"""
combined_ranking.py

CHECKPOINT 2 of the pipeline. Runs AFTER Caliby sequence design on the
backbones that survived interface_filters.py. Merges:

  1. Pre-Caliby interface geometry (from interface_metrics.csv)
  2. Caliby U-score (from seq_des_outputs.csv or score_outputs.csv)
  3. (optional) AF3 ipTM/PAE if you've run structure prediction on the
     Caliby-designed sequences for the top candidates

...into one ranked table, so you're not ranking on U-score alone -- a
sequence can have a great U-score (good fit to the backbone Caliby was
given) on a backbone that was geometrically borderline to start with.

Usage:
    python combined_ranking.py \
        --interface_csv interface_metrics.csv \
        --caliby_csv seq_des_outputs.csv \
        --out_csv combined_ranked.csv \
        [--af3_csv af3_iptm_results.csv]

Expected join key: both CSVs need a column that maps back to the same
backbone/example identifier. Adjust JOIN_KEY_INTERFACE / JOIN_KEY_CALIBY
below to match your actual column names -- Caliby's output uses
"example_id" or "out_pdb" depending on which script produced it
(seq_des.py vs score.py), and your interface_metrics.csv uses "file".
"""

import argparse
import csv
import re
from pathlib import Path

import pandas as pd

JOIN_KEY_INTERFACE = "file"        # column in interface_metrics.csv
JOIN_KEY_CALIBY = "out_pdb"        # column in Caliby's seq_des_outputs.csv / score_outputs.csv
                                     # (use "example_id" instead if that's what your run produced)


def normalize_id(s):
    """
    Strip extensions and common suffixes so the two CSVs join correctly even
    if one says 'trimer_motif_003.pdb' and the other says 'trimer_motif_003'
    or 'trimer_motif_003_seqdes0.pdb'. Adjust this regex if your naming
    convention differs.
    """
    s = str(s)
    s = re.sub(r"\.(pdb|cif)$", "", s)
    s = re.sub(r"_seqdes\d+$", "", s)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interface_csv", required=True)
    ap.add_argument("--caliby_csv", required=True)
    ap.add_argument("--af3_csv", default=None,
                     help="Optional: columns should include an id column plus "
                          "ipTM_AB, ipTM_BC, ipTM_CA (per-pair, not just global ipTM)")
    ap.add_argument("--out_csv", default="combined_ranked.csv")
    ap.add_argument("--top_n", type=int, default=10,
                     help="How many top candidates to print to stdout")
    args = ap.parse_args()

    iface = pd.read_csv(args.interface_csv)
    caliby = pd.read_csv(args.caliby_csv)

    iface["join_id"] = iface[JOIN_KEY_INTERFACE].apply(normalize_id)
    caliby["join_id"] = caliby[JOIN_KEY_CALIBY].apply(normalize_id)

    merged = pd.merge(iface, caliby, on="join_id", how="inner", suffixes=("_iface", "_caliby"))
    n_dropped = max(len(iface), len(caliby)) - len(merged)
    if n_dropped > 0:
        print(f"WARNING: {n_dropped} rows did not join between the two CSVs. "
              f"Check JOIN_KEY_INTERFACE/JOIN_KEY_CALIBY and the normalize_id() "
              f"regex against your actual filenames.")

    if args.af3_csv:
        af3 = pd.read_csv(args.af3_csv)
        af3["join_id"] = af3.iloc[:, 0].apply(normalize_id)  # assumes first col is the id
        merged = pd.merge(merged, af3, on="join_id", how="left")
        # decompose: flag if any single pairwise ipTM is much worse than the others
        # (catches the case your global ipTM would hide -- one weak interface
        # dragged up by two strong ones)
        iptm_cols = [c for c in merged.columns if c.lower().startswith("iptm_")]
        if iptm_cols:
            merged["iptm_min"] = merged[iptm_cols].min(axis=1)
            merged["iptm_spread"] = merged[iptm_cols].max(axis=1) - merged[iptm_cols].min(axis=1)

    # Only rank structures that passed the geometry checkpoint
    merged = merged[merged["status"] == "PASS"].copy()

    # --- composite score ---
    # Lower is better for: U (Caliby energy), bsa_spread, com_angle_max_dev, motif_symm_rmsd
    # We z-score each component (within this pool) so they're comparable, then sum.
    # This is a simple starting heuristic -- once you have real BSA/dG from a proper
    # SASA tool, swap the geometry terms for those instead of the contact-count proxy.
    def zscore(col):
        return (merged[col] - merged[col].mean()) / (merged[col].std(ddof=0) + 1e-9)

    components = []
    if "U" in merged.columns:
        merged["z_U"] = zscore("U")
        components.append("z_U")
    if "bsa_spread" in merged.columns:
        merged["z_bsa_spread"] = zscore("bsa_spread")
        components.append("z_bsa_spread")
    if "com_angle_max_dev" in merged.columns:
        merged["z_com_dev"] = zscore("com_angle_max_dev")
        components.append("z_com_dev")
    if "motif_symm_rmsd" in merged.columns:
        merged["z_symm_rmsd"] = zscore("motif_symm_rmsd")
        components.append("z_symm_rmsd")
    if "iptm_min" in merged.columns:
        # higher ipTM is better, so flip sign before adding to a "lower is better" composite
        merged["z_neg_iptm_min"] = -zscore("iptm_min")
        components.append("z_neg_iptm_min")

    if components:
        merged["composite_score"] = merged[components].sum(axis=1)
        merged = merged.sort_values("composite_score")
    else:
        print("WARNING: no scoring components found -- check column names in the merged table.")

    merged.to_csv(args.out_csv, index=False)
    print(f"\nWrote {len(merged)} ranked candidates to {args.out_csv}")

    if components:
        display_cols = ["join_id", "composite_score"] + [c.replace("z_", "") for c in
                         ["U", "bsa_spread", "com_angle_max_dev", "motif_symm_rmsd", "iptm_min"]
                         if c.replace("z_", "") in merged.columns or c in merged.columns]
        display_cols = [c for c in display_cols if c in merged.columns]
        print(f"\nTop {args.top_n} candidates:")
        print(merged[display_cols].head(args.top_n).to_string(index=False))


if __name__ == "__main__":
    main()
