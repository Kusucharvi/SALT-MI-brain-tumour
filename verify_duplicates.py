#!/usr/bin/env python3
"""Visual check of the BRISC-vs-figshare duplicate detection: saves side-by-side pairs
(BRISC slice | its most similar figshare slice) for random flagged and kept slices, plus a CSV of all matches.

Usage: python verify_duplicates.py --cache results_v2/cache.npz --brisc_cache results_ext/brisc_cache.npz --out results_ext
"""
import argparse, os, sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_experiments as RE   # noqa: E402
from run_external import thumbs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--brisc_cache", required=True)
    ap.add_argument("--out", default="results_ext")
    ap.add_argument("--thresh", type=float, default=0.95)
    ap.add_argument("--n", type=int, default=12)
    args = ap.parse_args()
    f_img, _, f_y, f_pid = RE.load_dataset("unused", args.cache)
    z = np.load(args.brisc_cache, allow_pickle=True)
    b_img, b_y, b_name = z["img"], z["y"], z["name"]
    tb = thumbs(b_img)
    tf = torch.cat([thumbs(f_img), thumbs(f_img[:, :, ::-1].copy())])
    best, arg = [], []
    for i in range(0, len(tb), 512):
        s = tb[i:i + 512] @ tf.T
        v, a = s.max(1); best.append(v); arg.append(a)
    best = torch.cat(best).numpy(); arg = torch.cat(arg).numpy()
    flipped = arg >= len(f_img); match = arg % len(f_img)
    with open(os.path.join(args.out, "duplicate_matches.csv"), "w") as fh:
        fh.write("brisc_file,brisc_class,max_ncc,figshare_index,figshare_patient,figshare_class,flipped\n")
        for i in range(len(b_y)):
            fh.write(f"{b_name[i]},{RE.CLASSES[b_y[i]]},{best[i]:.4f},{match[i]},{f_pid[match[i]]},"
                     f"{RE.CLASSES[f_y[match[i]]]},{int(flipped[i])}\n")
    dup = best >= args.thresh
    same_class = (b_y[dup] == f_y[match[dup]]).mean() if dup.any() else float("nan")
    print(f"flagged {dup.sum()} / {len(dup)}; NCC >= 0.99: {(best >= 0.99).sum()}; "
          f"0.90 <= NCC < 0.99: {((best >= 0.90) & (best < 0.99)).sum()}; "
          f"distinct figshare slices matched: {len(np.unique(match[dup]))}; "
          f"class label agrees with matched figshare slice: {100 * same_class:.1f}%")
    rng = np.random.default_rng(0)
    pick = list(rng.choice(np.where(dup)[0], min(args.n, dup.sum()), replace=False))
    pick += list(rng.choice(np.where(~dup)[0], min(4, (~dup).sum()), replace=False))
    fig, ax = plt.subplots(len(pick), 2, figsize=(4.4, 2.2 * len(pick)), dpi=150, squeeze=False)
    for r, i in enumerate(pick):
        fi = f_img[match[i]].astype(np.float32)
        if flipped[i]:
            fi = fi[:, ::-1]
        ax[r, 0].imshow(b_img[i].astype(np.float32), cmap="gray"); ax[r, 0].set_title(f"BRISC ({RE.CLASSES[b_y[i]]})", fontsize=7)
        ax[r, 1].imshow(fi, cmap="gray")
        ax[r, 1].set_title(f"figshare ({RE.CLASSES[f_y[match[i]]]}), NCC {best[i]:.3f}" + (" flipped" if flipped[i] else ""), fontsize=7)
        for a in ax[r]: a.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "FigureE2_duplicate_pairs.png")); plt.close(fig)
    print("Saved FigureE2_duplicate_pairs.png and duplicate_matches.csv")


if __name__ == "__main__":
    main()
