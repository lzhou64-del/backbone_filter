"""
interface_filters.py

Extends the existing filter_backbones.py with chain-chain INTERFACE metrics,
treating each pairwise interface (A:B, B:C, C:A) of the C3 homotrimer as a
"binding interface" to be scored the same way you'd score an antibody:antigen
or binder:target interface.

Designed to slot in as CHECKPOINT 1 — runs on raw Protpardelle backbone PDBs,
BEFORE Caliby sequence design, so cheap geometric filtering happens first.

Dependencies: biopython, numpy
    pip install biopython numpy --break-system-packages

Usage:
    python interface_filters.py --pdb_dir outputs_nterm_motif/ --out_csv interface_metrics.csv
"""

import argparse
import csv
import itertools
import os
import warnings
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning

warnings.simplefilter("ignore", PDBConstructionWarning)

# ----------------------------------------------------------------------
# Tunable thresholds — adjust based on what you observe across your pool
# ----------------------------------------------------------------------
BSA_MIN = 600.0           # Angstrom^2, per pairwise interface
BSA_MAX = 3000.0          # flag overpacked/clumped interfaces
SYMM_RMSD_MAX = 2.0       # Angstrom, motif RMSD across the 3 chain copies
COM_ANGLE_TARGET = 120.0  # degrees, ideal C3 spacing
COM_ANGLE_TOL = 15.0      # degrees, allowed deviation from 120
CLASH_CUTOFF = 3.5        # Angstrom, CB-CB steric clash distance
CONTACT_CUTOFF = 8.0      # Angstrom, CA-CA distance defining "interface residue"


def load_structure(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(Path(pdb_path).stem, pdb_path)
    return structure[0]  # first model


def get_ca_coords(chain):
    coords, resids = [], []
    for res in chain:
        if "CA" in res:
            coords.append(res["CA"].coord)
            resids.append(res.id[1])
    return np.array(coords), resids


def get_cb_coords(chain):
    """Use CB, fall back to CA for glycine."""
    coords, resids = [], []
    for res in chain:
        if "CB" in res:
            coords.append(res["CB"].coord)
            resids.append(res.id[1])
        elif "CA" in res:
            coords.append(res["CA"].coord)
            resids.append(res.id[1])
    return np.array(coords), resids


def center_of_mass(chain):
    coords = np.array([atom.coord for atom in chain.get_atoms()])
    return coords.mean(axis=0)


def com_angle_deviation(model):
    """
    For a C3 trimer, the angle subtended at the global centroid by any two
    chain centers-of-mass should be ~120 degrees. Returns max deviation
    from 120 across the three angles, plus the three raw angles.
    """
    chains = list(model)
    if len(chains) != 3:
        return None, []

    coms = [center_of_mass(c) for c in chains]
    global_centroid = np.mean(coms, axis=0)

    vecs = [com - global_centroid for com in coms]
    angles = []
    for v1, v2 in itertools.combinations(vecs, 2):
        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_theta))
        angles.append(angle)

    angles = [float(a) for a in angles]  # cast off numpy scalar types for clean CSV output
    max_dev = max(abs(a - COM_ANGLE_TARGET) for a in angles)
    return float(max_dev), angles


def interface_residues(chain_a, chain_b, cutoff=CONTACT_CUTOFF):
    """Residues in chain_a within cutoff (CA-CA) of any residue in chain_b."""
    ca_a, resids_a = get_ca_coords(chain_a)
    ca_b, resids_b = get_ca_coords(chain_b)
    if len(ca_a) == 0 or len(ca_b) == 0:
        return set(), set()

    dists = np.linalg.norm(ca_a[:, None, :] - ca_b[None, :, :], axis=2)
    a_idx, b_idx = np.where(dists < cutoff)
    a_contact_resids = {resids_a[i] for i in a_idx}
    b_contact_resids = {resids_b[i] for i in b_idx}
    return a_contact_resids, b_contact_resids


def approximate_bsa(chain_a, chain_b, contact_cutoff=CONTACT_CUTOFF):
    """
    Cheap proxy for buried surface area without a full SASA calculation
    (no DSSP/FreeSASA dependency). Counts CA-CA contacts within cutoff and
    scales by an empirical per-contact area. This is NOT a substitute for
    a real BSA calc (e.g. FreeSASA complex-vs-isolated difference) before
    you commit to ordering constructs -- treat this as a fast relative
    ranking signal across your ~100 backbones, then run a real SASA tool
    (see note in main()) on the survivors.
    """
    a_contacts, b_contacts = interface_residues(chain_a, chain_b, contact_cutoff)
    n_contacts = len(a_contacts) + len(b_contacts)
    # Empirical scaling factor (~15 A^2 per contacting residue side) --
    # calibrate this against a real SASA tool on a handful of structures.
    return n_contacts * 15.0, a_contacts, b_contacts


def cb_clash_count(chain_a, chain_b, clash_cutoff=CLASH_CUTOFF):
    cb_a, _ = get_cb_coords(chain_a)
    cb_b, _ = get_cb_coords(chain_b)
    if len(cb_a) == 0 or len(cb_b) == 0:
        return 0
    dists = np.linalg.norm(cb_a[:, None, :] - cb_b[None, :, :], axis=2)
    return int(np.sum(dists < clash_cutoff))


def motif_symmetry_rmsd(model, motif_len):
    """
    RMSD of the N-terminal motif backbone across the 3 chains after optimal
    superposition. Reuses the logic from your existing filter_backbones.py
    symmetry check -- included here so this script can run standalone.
    """
    from Bio.PDB.Superimposer import Superimposer

    chains = list(model)
    if len(chains) != 3:
        return None

    motif_atoms = []
    for chain in chains:
        atoms = []
        for res in chain:
            if res.id[1] <= motif_len and "CA" in res:
                atoms.append(res["CA"])
        motif_atoms.append(atoms)

    if any(len(a) != len(motif_atoms[0]) for a in motif_atoms):
        return None  # motif length mismatch across chains -- structural problem itself

    sup = Superimposer()
    rmsds = []
    ref = motif_atoms[0]
    for other in motif_atoms[1:]:
        sup.set_atoms(ref, other)
        rmsds.append(sup.rms)
    return max(rmsds) if rmsds else None


def score_structure(pdb_path, motif_len):
    model = load_structure(pdb_path)
    chains = list(model)

    row = {"file": Path(pdb_path).name, "n_chains": len(chains)}

    if len(chains) != 3:
        row["status"] = "FAIL_chain_count"
        return row

    # --- pairwise interface metrics (A:B, B:C, C:A) ---
    pair_labels = ["A:B", "B:C", "C:A"]
    pairs = [(chains[0], chains[1]), (chains[1], chains[2]), (chains[2], chains[0])]

    bsa_values, clash_values = [], []
    for label, (c1, c2) in zip(pair_labels, pairs):
        bsa, _, _ = approximate_bsa(c1, c2)
        clashes = cb_clash_count(c1, c2)
        row[f"bsa_{label}"] = round(bsa, 1)
        row[f"clashes_{label}"] = clashes
        bsa_values.append(bsa)
        clash_values.append(clashes)

    # symmetry of interface AREA across the 3 pairs (not just motif RMSD) --
    # a true C3 interface should have near-identical BSA at all 3 junctions
    bsa_spread = max(bsa_values) - min(bsa_values)
    row["bsa_spread"] = round(bsa_spread, 1)
    row["bsa_mean"] = round(np.mean(bsa_values), 1)

    # --- global symmetry checks ---
    com_dev, angles = com_angle_deviation(model)
    row["com_angle_max_dev"] = round(com_dev, 2) if com_dev is not None else None
    row["com_angles"] = [round(a, 1) for a in angles]

    symm_rmsd = motif_symmetry_rmsd(model, motif_len)
    row["motif_symm_rmsd"] = round(symm_rmsd, 3) if symm_rmsd is not None else None

    # --- pass/fail against thresholds ---
    reasons = []
    if any(b < BSA_MIN for b in bsa_values):
        reasons.append("bsa_too_low")
    if any(b > BSA_MAX for b in bsa_values):
        reasons.append("bsa_too_high")
    if any(c > 0 for c in clash_values):
        reasons.append("steric_clash")
    if com_dev is not None and com_dev > COM_ANGLE_TOL:
        reasons.append("com_angle_off")
    if symm_rmsd is not None and symm_rmsd > SYMM_RMSD_MAX:
        reasons.append("motif_asymmetric")
    if symm_rmsd is None:
        reasons.append("motif_length_mismatch")

    row["status"] = "PASS" if not reasons else "FAIL_" + "+".join(reasons)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb_dir", required=True, help="Directory of Protpardelle output PDBs")
    ap.add_argument("--out_csv", default="interface_metrics.csv")
    ap.add_argument("--motif_len", type=int, default=24,
                     help="Length of the N-terminal trimerization motif (residues 1..motif_len)")
    args = ap.parse_args()

    pdb_files = sorted(Path(args.pdb_dir).glob("*.pdb"))
    print(f"Scoring {len(pdb_files)} structures from {args.pdb_dir} ...")

    rows = []
    for pdb_path in pdb_files:
        try:
            row = score_structure(pdb_path, args.motif_len)
        except Exception as e:
            row = {"file": pdb_path.name, "status": f"ERROR_{type(e).__name__}", "error_msg": str(e)}
        rows.append(row)

    fieldnames = sorted(set().union(*[set(r.keys()) for r in rows]))
    # keep a sensible column order: file, status, then the rest
    ordered = ["file", "status"] + [f for f in fieldnames if f not in ("file", "status")]

    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(rows)

    n_pass = sum(1 for r in rows if r.get("status") == "PASS")
    print(f"Wrote {args.out_csv}: {n_pass}/{len(rows)} passed all filters.")
    print("\nNOTE: approximate_bsa() is a fast contact-count proxy, not a real SASA-based")
    print("BSA calculation. Before ordering constructs, run survivors through a proper")
    print("tool (e.g. FreeSASA on isolated vs. complexed chains, or Rosetta InterfaceAnalyzer)")
    print("to get publication-grade BSA/dG numbers -- this script is for fast triage across")
    print("the full ~100-structure pool, not final validation.")


if __name__ == "__main__":
    main()
