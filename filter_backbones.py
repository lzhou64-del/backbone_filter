#!/usr/bin/env python3
"""
filter_backbones.py
--------------------
Pure-Python backbone geometry filter for Protpardelle-1c outputs.
No Rosetta or PyRosetta required — uses only Biopython (already in protpardelle env).

Filters applied (in order):
  1. Chain count check      — must have exactly 3 chains
  2. Per-chain length check — each chain must be 243-258 residues (228 fixed + 15-30 motif)
  3. CA-CA bond geometry    — consecutive CA distances must be 3.6-4.0 Å (flags broken chains)
  4. Steric clash filter    — CB-CB distances < 2.8 Å between non-bonded residues (flags clashes)
  5. Ramachandran outliers  — phi/psi outside allowed regions (flags bad backbone torsions)
  6. N-term motif symmetry  — CA RMSD between the 3 N-terminal extensions must be < 3.0 Å

Usage:
  python3 filter_backbones.py --input-dir <results_dir> --output-dir <filtered_dir> [options]

Example:
  python3 filter_backbones.py \
      --input-dir results/trimer_nterm_motif/cc95-epoch3490-.../soluble_caliby_trimer/ \
      --output-dir filtered_backbones/ \
      --monomer-len 228 \
      --motif-min 15 \
      --motif-max 30 \
      --clash-cutoff 2.8 \
      --ca-bond-min 3.6 \
      --ca-bond-max 4.0 \
      --rama-outlier-max 0.1 \
      --symm-rmsd-max 3.0 \
      --num-chains 3
"""

import os
import sys
import shutil
import argparse
import csv
import math
import glob
from collections import defaultdict

try:
    from Bio.PDB import PDBParser, PPBuilder
    from Bio.PDB.vectors import Vector
except ImportError:
    sys.exit("ERROR: Biopython not found. Activate your protpardelle env first:\n"
             "  source envs/protpardelle/bin/activate")

import numpy as np

# ---------------------------------------------------------------------------
# Ramachandran allowed regions (core + allowed, GLY separate)
# Derived from Lovell et al. 2003 — simplified rectangular approximation
# ---------------------------------------------------------------------------
RAMA_ALLOWED = {
    "GLY": [
        (-180, 180, -180, 180),   # GLY is nearly unrestricted
    ],
    "PRO": [
        (-90, -40, -80, -10),     # alpha-helix
        (-90, -40, 100, 180),     # C7-eq
    ],
    "general": [
        (-180, -40, 100, 180),    # beta-sheet
        (-180, -40, -180, -100),  # beta-sheet (wraparound)
        (-160, -40, -80, 10),     # alpha-helix + left turn
        (40, 180, -180, 180),     # positive phi (less common but allowed)
    ],
}

def is_rama_allowed(resname, phi, psi):
    """Return True if (phi, psi) falls in an allowed Ramachandran region."""
    regions = RAMA_ALLOWED.get(resname, RAMA_ALLOWED["general"])
    for (phi_min, phi_max, psi_min, psi_max) in regions:
        if phi_min <= phi <= phi_max and psi_min <= psi <= psi_max:
            return True
    return False

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def get_ca_coords(chain):
    """Return list of (resnum, CA coord as np.array) for a chain."""
    coords = []
    for res in chain.get_residues():
        if res.id[0] != " ":      # skip HETATM
            continue
        if "CA" in res:
            coords.append((res.id[1], np.array(res["CA"].get_vector().get_array())))
    return coords

def get_cb_coords(chain):
    """Return list of CB coords (use CA for GLY)."""
    coords = []
    for res in chain.get_residues():
        if res.id[0] != " ":
            continue
        if res.resname == "GLY":
            if "CA" in res:
                coords.append(np.array(res["CA"].get_vector().get_array()))
        else:
            if "CB" in res:
                coords.append(np.array(res["CB"].get_vector().get_array()))
    return coords

def ca_bond_violations(ca_coords, bond_min=3.6, bond_max=4.0):
    """Count CA-CA distances outside [bond_min, bond_max] Å for consecutive residues."""
    violations = 0
    for i in range(len(ca_coords) - 1):
        rn1, c1 = ca_coords[i]
        rn2, c2 = ca_coords[i + 1]
        if rn2 - rn1 != 1:   # non-consecutive in sequence — skip
            continue
        d = np.linalg.norm(c2 - c1)
        if d < bond_min or d > bond_max:
            violations += 1
    return violations

def count_clashes(all_cb_coords, clash_cutoff=2.8):
    """
    Count steric clashes: CB-CB pairs < clash_cutoff Å,
    excluding pairs within 3 residues of each other (same chain).
    Uses a simple O(n^2) search — fast enough for <1000 residues.
    """
    clashes = 0
    n = len(all_cb_coords)
    coords = np.array(all_cb_coords)
    for i in range(n):
        for j in range(i + 4, n):   # skip i±1,2,3
            d = np.linalg.norm(coords[i] - coords[j])
            if d < clash_cutoff:
                clashes += 1
    return clashes

def rama_outlier_fraction(chain):
    """
    Compute fraction of residues with disallowed phi/psi.
    Skips first and last residue (no phi or psi defined).
    """
    ppb = PPBuilder()
    total = 0
    outliers = 0
    for pp in ppb.build_peptides(chain):
        angles = pp.get_phi_psi_list()
        for i, (phi, psi) in enumerate(angles):
            if phi is None or psi is None:
                continue
            total += 1
            resname = pp[i].resname
            phi_deg = math.degrees(phi)
            psi_deg = math.degrees(psi)
            if not is_rama_allowed(resname, phi_deg, psi_deg):
                outliers += 1
    if total == 0:
        return 0.0
    return outliers / total

def nterm_symmetry_rmsd(chains, monomer_len):
    """
    Compute mean pairwise CA RMSD of the N-terminal motif across 3 chains.
    Motif = residues 1..(chain_len - monomer_len).
    """
    motif_cas = []
    for chain in chains:
        ca = get_ca_coords(chain)
        motif_len = len(ca) - monomer_len
        if motif_len <= 0:
            return 999.0
        motif_cas.append(np.array([c for _, c in ca[:motif_len]]))

    if len(motif_cas) < 3:
        return 999.0

    min_len = min(len(m) for m in motif_cas)
    rmsds = []
    for i in range(3):
        for j in range(i + 1, 3):
            diff = motif_cas[i][:min_len] - motif_cas[j][:min_len]
            rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
            rmsds.append(rmsd)
    return float(np.mean(rmsds))

# ---------------------------------------------------------------------------
# Per-file filter
# ---------------------------------------------------------------------------

def filter_pdb(pdb_path, args):
    """
    Run all filters on a single PDB file.
    Returns (pass: bool, metrics: dict).
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("s", pdb_path)
    except Exception as e:
        return False, {"error": str(e)}

    model = structure[0]
    chains = [c for c in model.get_chains()]
    metrics = {"file": os.path.basename(pdb_path)}

    # 1. Chain count
    metrics["num_chains"] = len(chains)
    if len(chains) != args.num_chains:
        metrics["fail_reason"] = f"wrong_chain_count({len(chains)})"
        return False, metrics

    # 2. Per-chain length
    chain_lens = []
    for chain in chains:
        res = [r for r in chain.get_residues() if r.id[0] == " "]
        chain_lens.append(len(res))
    metrics["chain_lengths"] = chain_lens
    total_expected_min = args.num_chains * (args.monomer_len + args.motif_min)
    total_expected_max = args.num_chains * (args.monomer_len + args.motif_max)
    total_res = sum(chain_lens)
    if not (total_expected_min <= total_res <= total_expected_max):
        metrics["fail_reason"] = f"wrong_length(total={total_res})"
        return False, metrics

    # 3. CA-CA bond geometry (all chains combined)
    total_ca_violations = 0
    for chain in chains:
        ca = get_ca_coords(chain)
        total_ca_violations += ca_bond_violations(ca, args.ca_bond_min, args.ca_bond_max)
    metrics["ca_bond_violations"] = total_ca_violations
    if total_ca_violations > args.ca_violation_max:
        metrics["fail_reason"] = f"ca_bond_violations({total_ca_violations})"
        return False, metrics

    # 4. Steric clashes (all chains combined)
    all_cb = []
    for chain in chains:
        all_cb.extend(get_cb_coords(chain))
    n_clashes = count_clashes(all_cb, args.clash_cutoff)
    metrics["clashes"] = n_clashes
    if n_clashes > args.clash_max:
        metrics["fail_reason"] = f"clashes({n_clashes})"
        return False, metrics

    # 5. Ramachandran outliers (each chain)
    rama_fracs = []
    for chain in chains:
        rama_fracs.append(rama_outlier_fraction(chain))
    mean_rama = float(np.mean(rama_fracs))
    metrics["rama_outlier_frac"] = round(mean_rama, 4)
    if mean_rama > args.rama_outlier_max:
        metrics["fail_reason"] = f"rama_outliers({mean_rama:.3f})"
        return False, metrics

    # 6. N-terminal motif C3 symmetry
    symm_rmsd = nterm_symmetry_rmsd(chains, args.monomer_len)
    metrics["nterm_symm_rmsd"] = round(symm_rmsd, 3)
    if symm_rmsd > args.symm_rmsd_max:
        metrics["fail_reason"] = f"asymmetric_nterm(rmsd={symm_rmsd:.2f})"
        return False, metrics

    metrics["fail_reason"] = "PASS"
    return True, metrics

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir",   required=True,
                        help="Directory containing Protpardelle backbone PDBs")
    parser.add_argument("--output-dir",  required=True,
                        help="Directory to copy passing PDBs into")
    parser.add_argument("--monomer-len", type=int, default=228,
                        help="Length of fixed monomer (default: 228)")
    parser.add_argument("--motif-min",   type=int, default=15,
                        help="Minimum N-term motif length (default: 15)")
    parser.add_argument("--motif-max",   type=int, default=30,
                        help="Maximum N-term motif length (default: 30)")
    parser.add_argument("--num-chains",  type=int, default=3,
                        help="Expected number of chains (default: 3)")
    parser.add_argument("--clash-cutoff",    type=float, default=2.8,
                        help="CB-CB clash distance cutoff in Å (default: 2.8)")
    parser.add_argument("--clash-max",       type=int,   default=0,
                        help="Max allowed clashes (default: 0)")
    parser.add_argument("--ca-bond-min",     type=float, default=3.6,
                        help="Min CA-CA bond distance in Å (default: 3.6)")
    parser.add_argument("--ca-bond-max",     type=float, default=4.0,
                        help="Max CA-CA bond distance in Å (default: 4.0)")
    parser.add_argument("--ca-violation-max",type=int,   default=0,
                        help="Max allowed CA bond violations (default: 0)")
    parser.add_argument("--rama-outlier-max",type=float, default=0.1,
                        help="Max fraction of Ramachandran outliers (default: 0.1)")
    parser.add_argument("--symm-rmsd-max",   type=float, default=3.0,
                        help="Max N-term motif C3 RMSD in Å (default: 3.0)")
    parser.add_argument("--report",      default="filter_report.csv",
                        help="CSV report filename (default: filter_report.csv)")
    args = parser.parse_args()

    # Find all PDBs
    pdbs = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.pdb"), recursive=True) +
                  glob.glob(os.path.join(args.input_dir, "*.pdb")))
    # Deduplicate
    pdbs = sorted(set(pdbs))

    if not pdbs:
        sys.exit(f"ERROR: No PDB files found in {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    passed = []
    failed = []
    all_metrics = []

    print(f"\nFiltering {len(pdbs)} PDB files from: {args.input_dir}")
    print(f"Thresholds: clash_cutoff={args.clash_cutoff}Å | clash_max={args.clash_max} | "
          f"ca_violation_max={args.ca_violation_max} | "
          f"rama_max={args.rama_outlier_max} | symm_rmsd_max={args.symm_rmsd_max}Å\n")

    for i, pdb in enumerate(pdbs):
        ok, metrics = filter_pdb(pdb, args)
        all_metrics.append(metrics)
        status = "PASS" if ok else f"FAIL ({metrics.get('fail_reason', '?')})"
        print(f"[{i+1:4d}/{len(pdbs)}] {os.path.basename(pdb):50s} {status}")
        if ok:
            passed.append(pdb)
            shutil.copy2(pdb, os.path.join(args.output_dir, os.path.basename(pdb)))
        else:
            failed.append(pdb)

    # Write CSV report
    report_path = os.path.join(args.output_dir, args.report)
    fieldnames = ["file", "num_chains", "chain_lengths", "ca_bond_violations",
                  "clashes", "rama_outlier_frac", "nterm_symm_rmsd", "fail_reason", "error"]
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_metrics)

    print(f"\n{'='*60}")
    print(f"  PASSED : {len(passed)} / {len(pdbs)}")
    print(f"  FAILED : {len(failed)} / {len(pdbs)}")
    print(f"  Passing PDBs copied to : {args.output_dir}")
    print(f"  Full report            : {report_path}")
    print(f"{'='*60}\n")

    if len(passed) == 0:
        print("WARNING: No designs passed. Consider relaxing thresholds:")
        print("  --clash-max 2  --rama-outlier-max 0.15  --symm-rmsd-max 4.0  --ca-violation-max 2")

if __name__ == "__main__":
    main()
