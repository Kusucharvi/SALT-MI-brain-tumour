#!/usr/bin/env python3
"""
External validation of explanation multiplicity on BRISC 2025.

1. Downloads BRISC 2025 from Zenodo (record 17524350, ~260 MB): T1-weighted slices with expert masks.
2. Removes BRISC slices that duplicate (near-identically) any slice of the figshare training dataset.
3. Trains ResNet-50 and DenseNet-121 with 5 seeds each on ALL figshare patients (85/15 patient split for
   early stopping) and saves the weights.
4. On the remaining BRISC tumour slices (glioma, meningioma, pituitary), measures accuracy, prediction
   disagreement and explanation multiplicity (Grad-CAM, Grad-CAM++, HiResCAM) exactly as in run_multiplicity.py.

Needs run_experiments.py and run_multiplicity.py in the same folder. Resumable.

Usage:
  python run_external.py --out results_ext --cache results_v2/cache.npz
"""
import argparse, glob, json, os, random, sys, time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_experiments as RE   # noqa: E402
import run_multiplicity as RM  # noqa: E402

CLASSES = RE.CLASSES
CODE = {"gl": 0, "me": 1, "pi": 2}          # 'nt' (no tumour) is excluded: the models are 3-class
ZENODO_RECORD = 17524350


# ------------------------------------------------------------------ BRISC data
def download_brisc(d):
    os.makedirs(d, exist_ok=True)
    if glob.glob(os.path.join(d, "**", "segmentation_task"), recursive=True):
        return
    print("Downloading BRISC 2025 from Zenodo ...")
    req = RE.urllib.request.Request(f"https://zenodo.org/api/records/{ZENODO_RECORD}",
                                    headers={"User-Agent": "Mozilla/5.0 (research script)"})
    rec = json.load(RE.urllib.request.urlopen(req, timeout=60))
    for fi in rec["files"]:
        if fi["key"].endswith(".zip"):
            dest = os.path.join(d, fi["key"])
            if not os.path.exists(dest):
                RE._fetch(fi["links"].get("content", fi["links"]["self"]), dest)
            print("  unzipping"); RE._safe_extract(dest, d); os.remove(dest)
    if not glob.glob(os.path.join(d, "**", "segmentation_task"), recursive=True):
        raise SystemExit("BRISC download/unzip failed. Download brisc2025.zip from https://zenodo.org/records/17524350 "
                         f"and unzip it into {d}")


def load_brisc(d, cache, size=256, mres=128):
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["img"], z["mask"], z["y"], z["name"], z["plane"]
    files = sorted(glob.glob(os.path.join(d, "**", "segmentation_task", "*", "images", "*.jpg"), recursive=True))
    imgs, masks, ys, names, planes = [], [], [], [], []
    for i, f in enumerate(files):
        parts = os.path.basename(f).split("_")          # brisc2025_<split>_<idx>_<tumor>_<view>_<seq>.jpg
        if len(parts) < 6 or parts[3] not in CODE:
            continue
        mp = f.replace(os.sep + "images" + os.sep, os.sep + "masks" + os.sep)[:-4] + ".png"
        if not os.path.exists(mp):
            continue
        im = np.asarray(Image.open(f).convert("L"), dtype=np.float32)
        mk = (np.asarray(Image.open(mp).convert("L")) > 127).astype(np.float32)
        if mk.sum() == 0:
            continue
        im = (im - im.min()) / (im.max() - im.min() + 1e-8)
        im = F.interpolate(torch.from_numpy(im)[None, None], size=(size, size), mode="bilinear", align_corners=False)[0, 0]
        mk = F.interpolate(torch.from_numpy(mk)[None, None], size=(512, 512), mode="nearest")
        mk = F.max_pool2d(mk, 512 // mres)[0, 0]
        imgs.append(im.numpy().astype(np.float16)); masks.append(mk.numpy().astype(np.uint8))
        ys.append(CODE[parts[3]]); names.append(os.path.basename(f)); planes.append(parts[4])
        if (i + 1) % 1000 == 0:
            print(f"  read {i + 1}/{len(files)}")
    img, mask, y = np.stack(imgs), np.stack(masks), np.array(ys)
    name, plane = np.array(names), np.array(planes)
    np.savez(cache, img=img, mask=mask, y=y, name=name, plane=plane)
    return img, mask, y, name, plane


def thumbs(img, n=32):
    t = F.interpolate(torch.from_numpy(np.asarray(img, dtype=np.float32))[:, None], size=(n, n), mode="area").flatten(1)
    t = t - t.mean(1, keepdim=True)
    return t / (t.norm(dim=1, keepdim=True) + 1e-8)


def find_duplicates(b_img, f_img, thresh, out):
    """Max normalised cross-correlation (32x32 thumbnails, incl. horizontally flipped) of each BRISC slice
    against every figshare slice."""
    tb = thumbs(b_img)
    tf = torch.cat([thumbs(f_img), thumbs(f_img[:, :, ::-1].copy())])
    best = torch.cat([(tb[i:i + 512] @ tf.T).amax(1) for i in range(0, len(tb), 512)]).numpy()
    fig, ax = plt.subplots(figsize=(5, 3.3), dpi=200)
    ax.hist(best, bins=60, color="#2F5D8A"); ax.axvline(thresh, color="#C0504D", ls="--", label=f"threshold {thresh}")
    ax.set_xlabel("Max similarity of BRISC slice to any figshare slice (NCC)"); ax.set_ylabel("BRISC slices")
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(out, "FigureE0_duplicate_check.png")); plt.close(fig)
    return best


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_ext")
    ap.add_argument("--cache", required=True, help="figshare cache.npz (e.g. results_v2/cache.npz)")
    ap.add_argument("--data_dir", default="data/figshare_brain_tumour")
    ap.add_argument("--brisc_dir", default="/content/brisc")
    ap.add_argument("--archs", nargs="+", default=["resnet50", "densenet121"], choices=["resnet50", "densenet121"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--dup_thresh", type=float, default=0.95)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cam_res", type=int, default=128)
    ap.add_argument("--no_pretrained", action="store_true", help="offline testing only")
    args = ap.parse_args()
    for sub in ("runs", "models"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logf = open(os.path.join(args.out, "log.txt"), "a")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    # ---- data
    f_img, _, f_y, f_pid = RE.load_dataset(args.data_dir, args.cache)
    bcache = os.path.join(args.out, "brisc_cache.npz")
    if not os.path.exists(bcache):
        download_brisc(args.brisc_dir)
    b_img, b_mask, b_y, b_name, b_plane = load_brisc(args.brisc_dir, bcache, mres=args.cam_res)
    log(f"figshare: {len(f_y)} slices / {len(np.unique(f_pid))} patients;  BRISC tumour slices with masks: {len(b_y)} "
        f"({', '.join(f'{c}: {(b_y == k).sum()}' for k, c in enumerate(CLASSES))})")
    best = find_duplicates(b_img, f_img, args.dup_thresh, args.out)
    keep = best < args.dup_thresh
    log(f"Duplicate check (NCC >= {args.dup_thresh}): {int((~keep).sum())} BRISC slices removed; {int(keep.sum())} kept "
        f"({', '.join(f'{c}: {int(((b_y == k) & ~keep).sum())} removed' for k, c in enumerate(CLASSES))})")
    idx = np.where(keep)[0]
    ds_b = RE.SliceDS(b_img, b_y, idx, False)
    dl_b = DataLoader(ds_b, batch_size=args.batch, num_workers=args.workers)

    # ---- training on all figshare patients
    pats = np.unique(f_pid); plab = np.array([np.bincount(f_y[f_pid == p]).argmax() for p in pats])
    tr_p, va_p = train_test_split(pats, test_size=0.15, stratify=plab, random_state=args.split_seed)
    tr = np.where(np.isin(f_pid, tr_p))[0]; va = np.where(np.isin(f_pid, va_p))[0]
    for arch in args.archs:
        for s in args.seeds:
            rp = os.path.join(args.out, "runs", f"{arch}_s{s}.npz")
            if os.path.exists(rp):
                log(f"[{arch} seed {s}] done already, skipping"); continue
            wp = os.path.join(args.out, "models", f"{arch}_s{s}.pt")
            if os.path.exists(wp):
                model, _ = RE.build_model(arch, False); model.load_state_dict(torch.load(wp, map_location="cpu")); model = model.to(dev)
                log(f"\n=== {arch} | seed {s} | loaded saved weights ===")
            else:
                log(f"\n=== {arch} | seed {s} | train {len(tr)} val {len(va)} figshare slices ===")
                t0 = time.time()
                model, n_ep, _ = RM.train_one(arch, tr, va, f_img, f_y, args, dev, log, s)
                torch.save(model.state_dict(), wp)
                log(f"    trained in {time.time() - t0:.0f}s ({n_ep} epochs)")
            probs, _ = RM.predict_with_features(model, arch, dl_b, dev)
            cams = RM.multi_cam_maps(model, arch, ds_b, probs.argmax(1), dev, args.cam_res)
            acc = float((probs.argmax(1) == b_y[idx]).mean())
            m = RE.metrics(b_y[idx], probs)
            log(f"    BRISC external accuracy {acc:.4f}  macroF1 {m['macro_f1']:.4f}")
            np.savez_compressed(rp, idx=idx, probs=probs.astype(np.float32), **{f"cam_{k}": v for k, v in cams.items()})
            json.dump(dict(accuracy=acc, macro_f1=m["macro_f1"], cm=m["cm"]), open(rp.replace(".npz", ".json"), "w"))
            del model; torch.cuda.empty_cache() if dev.type == "cuda" else None

    # ---- analysis
    L = ["EXTERNAL VALIDATION ON BRISC 2025 — RESULTS SUMMARY",
         f"Models trained on all {len(np.unique(f_pid))} figshare patients; evaluated on {len(idx)} BRISC tumour slices "
         f"after removing {int((~keep).sum())} near-duplicates of figshare slices (NCC >= {args.dup_thresh}).",
         "BRISC has no patient identifiers: CIs below resample slices, not patients.", ""]
    masks = b_mask[idx]; yt = b_y[idx]; planes = b_plane[idx]
    maps_g = {}
    for arch in args.archs:
        runs = [s for s in args.seeds if os.path.exists(os.path.join(args.out, "runs", f"{arch}_s{s}.npz"))]
        if len(runs) < 2:
            continue
        D = [np.load(os.path.join(args.out, "runs", f"{arch}_s{s}.npz")) for s in runs]
        P = np.stack([d["probs"] for d in D]); pred = P.argmax(2)
        accs = (pred == yt).mean(1)
        f1s = [json.load(open(os.path.join(args.out, "runs", f"{arch}_s{s}.json")))["macro_f1"] for s in runs]
        disagree = (pred != pred[0]).any(0); allc = (pred == yt).all(0)
        L.append(f"=== {arch} === accuracy per seed: " + ", ".join(f"{100 * a:.1f}" for a in accs) +
                 f"  (mean {100 * accs.mean():.1f} ± {100 * accs.std():.1f}); macro F1 {100 * np.mean(f1s):.1f} ± {100 * np.std(f1s):.1f}")
        L.append(f"  prediction disagreement between seeds: {100 * disagree.mean():.1f}%;  slices correct in every seed: {int(allc.sum())}")
        per_method = {}
        if allc.sum() < 2:
            L.append("  too few slices correct in every seed for explanation analysis"); L.append(""); continue
        L.append("  method | hit rate % | hit flips % | hit by all seeds % | median map r | top-20% IoU | peak distance mm | ensemble hit % | energy (single / ensemble)")
        for m in RM.METHODS:
            C = np.stack([d[f"cam_{m}"].astype(np.float32) for d in D])
            r = RM.multiplicity_block(C, masks, allc, args.cam_res)
            r["_slice"]["pid"] = np.where(allc)[0]                     # slice-level resampling (no patient IDs)
            per_method[m] = [r]
            L.append(f"  {m} | {100 * r['hit']:.1f} | {100 * r['hit_flip']:.1f} | {100 * r['all_hit']:.1f} | {r['pear']:.3f} | "
                     f"{r['tiou']:.3f} | {r['peak_mm']:.1f} | {100 * r['ens_hit']:.1f} | {r['energy']:.3f} / {r['ens_energy']:.3f}")
            if m == "gradcam":
                maps_g[arch] = (C, allc)
                for pl in ("ax", "co", "sa"):
                    sel = allc & (planes == pl)
                    if sel.sum():
                        hp = np.stack([RM.loc_metrics(C[s_], masks)[0] for s_ in range(len(C))])[:, sel]
                        L.append(f"    Grad-CAM by plane {pl}: n={int(sel.sum())}  hit {100 * hp.mean():.1f}%  "
                                 f"flips {100 * (hp.any(0) & ~hp.all(0)).mean():.1f}%")
                if arch == args.archs[0] and len(runs) >= 3:
                    j = np.where(allc)[0][np.argmin(r["_pear_all"][allc])]
                    RM.fig_seeds(b_img[idx[j]], C[:, j], r["_ens"][j], masks[j], runs, r["_hits"][:, j],
                                 os.path.join(args.out, "FigureE1_seed_heatmaps.png"))
        L += [x.replace("pooled over folds, ", "").replace("patient-level bootstrap", "slice-level bootstrap").replace("n slices / patients", "n slices / n slices")
              for x in RM.ci_table(per_method, args)]
        L.append("")
    if len(maps_g) == 2:
        (C0, a0), (C1, a1) = maps_g[args.archs[0]], maps_g[args.archs[1]]
        both = a0 & a1; n = min(len(C0), len(C1))
        if both.sum() > 1:
            r = np.median(np.mean([RM.pearson_rows(C0[i][both].reshape(both.sum(), -1), C1[i][both].reshape(both.sum(), -1)) for i in range(n)], 0))
            d = np.median(np.mean([np.linalg.norm(RM.peak_xy(C0[i][both]) - RM.peak_xy(C1[i][both]), axis=1) for i in range(n)], 0)) * (512 / args.cam_res * RM.PIX_MM)
            L.append(f"Cross-architecture Grad-CAM agreement: median r {r:.3f}, peak distance {d:.1f} mm (pixel size assumed 0.49 mm at 512 px)")
    open(os.path.join(args.out, "external_summary.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    log(f"\nDone. Send {os.path.join(args.out, 'external_summary.txt')} and the FigureE*.png files.")


if __name__ == "__main__":
    main()
