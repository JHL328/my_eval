#!/usr/bin/env python3
"""
Plot bbq_ablations PDFs directly from the per-task CSVs (Model rows x step columns).

We plot from the CSVs (not passk.json) because the existing-mix rows in the CSVs
aggregate several checkpoint series whose raw names are no longer 1:1 recoverable
from the cumulative passk.json. Plotting from the CSV guarantees the figure matches
the table exactly: existing mixes + the two new bbq-ablation lines.

Usage:
  python plot_bbq_from_csv.py --root <bbq_ablations dir> [--subdir base math code]
"""
import argparse
import os
import glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

# Models to draw thicker / on top so the new ablation lines stand out.
HIGHLIGHT = {"mix-bbq-all-mask", "mix-bbq-all-baseline"}

# Progression CSVs are produced by the puzzles workflow; skip them here.
SKIP_SUFFIX = "_tokenmix_progression"


def csv_to_pdf_name(csv_basename, subdir):
    stem = csv_basename[:-4] if csv_basename.endswith(".csv") else csv_basename
    # strip lm-eval metric qualifiers
    for suf in (",none", ",remove_whitespace"):
        stem = stem.replace(suf, "")
    # ifeval table -> simple name to match the existing PDF
    if stem.startswith("ifeval_"):
        stem = "ifeval"
    # code PDFs use passN (no @)
    if subdir == "code":
        stem = stem.replace("pass@", "pass")
    return stem + ".pdf"


def plot_one(csv_path, pdf_path):
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=True)
    if "Model" not in df.columns or len(df) == 0:
        print(f"  skip (no data): {csv_path}")
        return False
    step_cols = []
    for c in df.columns:
        if c == "Model":
            continue
        try:
            int(c)
            step_cols.append(c)
        except (ValueError, TypeError):
            pass
    if not step_cols:
        print(f"  skip (no step columns): {csv_path}")
        return False
    steps_int = sorted(int(c) for c in step_cols)

    models = list(df["Model"])
    cmap = cm.get_cmap("tab20", max(len(models), 1))

    plt.figure(figsize=(16, 7), facecolor="#f7f7f7")
    drawn = 0
    # draw non-highlight first, highlight last (on top)
    order = [m for m in models if m not in HIGHLIGHT] + [m for m in models if m in HIGHLIGHT]
    for i, model in enumerate(order):
        row = df[df["Model"] == model].iloc[0]
        xs, ys = [], []
        for s in steps_int:
            v = row.get(str(s))
            if v is not None and str(v).strip() != "" and not pd.isna(v):
                xs.append(s)
                ys.append(float(v))
        if not xs:
            continue
        hl = model in HIGHLIGHT
        plt.plot(
            xs, ys,
            marker="o" if not hl else "D",
            label=model,
            color=cmap(models.index(model)),
            linewidth=3.0 if hl else 1.8,
            markersize=9 if hl else 6,
            zorder=5 if hl else 3,
        )
        drawn += 1

    if drawn == 0:
        plt.close()
        print(f"  skip (no plottable rows): {csv_path}")
        return False

    title = os.path.basename(pdf_path)[:-4]
    plt.xlabel("Checkpoint step", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.title(f"{title}  (bbq ablations)", fontsize=16, weight="bold")
    plt.legend(title="Mix", loc="center left", bbox_to_anchor=(1.01, 0.5),
               fontsize=10, title_fontsize=12, frameon=False, borderaxespad=0.)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(pdf_path, format="pdf", bbox_inches="tight", facecolor="#f7f7f7")
    plt.close()
    print(f"  saved {pdf_path} ({drawn} lines)")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="bbq_ablations directory")
    ap.add_argument("--subdir", nargs="+", default=["base", "math", "code"])
    args = ap.parse_args()

    total = 0
    for sub in args.subdir:
        d = os.path.join(args.root, sub)
        if not os.path.isdir(d):
            continue
        print(f"=== {sub}/ ===")
        for csv_path in sorted(glob.glob(os.path.join(d, "*.csv"))):
            base = os.path.basename(csv_path)
            if base[:-4].endswith(SKIP_SUFFIX):
                continue
            pdf_path = os.path.join(d, csv_to_pdf_name(base, sub))
            if plot_one(csv_path, pdf_path):
                total += 1
    print(f"\nDone. Generated/updated {total} PDFs.")


if __name__ == "__main__":
    main()
