#!/usr/bin/env python3
"""
Independent external validation on the Medical Segmentation Decathlon brain tumour task (Task01_BrainTumour,
derived from BraTS 2016/2017; CC-BY-SA 4.0): 484 multi-parametric MRI volumes of gliomas with expert labels.

Uses the 10 models saved by run_external.py (ResNet-50 and DenseNet-121, 5 seeds each, trained on all figshare
patients). From each volume the contrast-enhanced T1 (T1gd) axial slices with the largest tumour-core area are
extracted (tumour core = labels 2 and 3: non-enhancing and enhancing tumour; oedema excluded, matching figshare
masks drawn on contrast-enhanced T1). All cases are gliomas.

Reports: glioma recall per seed, prediction disagreement, seed dependence of Grad-CAM / Grad-CAM++ / HiResCAM and
occlusion against the masks, and the image-centre baseline, with patient-level bootstrap CIs.

Needs run_experiments.py, run_multiplicity.py and run_occlusion.py in the same folder. Resumable.

Usage:
  pip install nibabel
  python run_external_msd.py --models results_ext/models --cache results_mult2/cache.npz --out results_msd
"""
import argparse, glob, json, os, sys, tarfile, time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_experiments as RE   # noqa: E402
import run_multiplicity as RM  # noqa: E402
import run_occlusion as RO     # noqa: E402

MSD_URL = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
T1GD = 2                    # channel order in Task01: 0 FLAIR, 1 T1w, 2 T1gd, 3 T2w
GLIOMA = 0


def download_msd(d):
    if glob.glob(os.path.join(d, "**", "imagesTr"), recursive=True):
        return
    os.makedirs(d, exist_ok=True)
    tp = os.path.join(d, "Task01_BrainTumour.tar")
    if not os.path.exists(tp):
        print("Downloading Task01_BrainTumour (~7 GB) from the AWS Open Data registry ...")
        RE._fetch(MSD_URL, tp)
    print("Extracting ...")
    with tarfile.open(tp) as t:
        try:
            t.extractall(d, filter="data")
        except TypeError:            # Python < 3.12
            root = os.path.realpath(d)
            for m in t.getmembers():
                if not os.path.realpath(os.path.join(d, m.name)).startswith(root):
                    raise SystemExit(f"Unsafe path in tar: {m.name}")
            t.extractall(d)
    os.remove(tp)


def thumb_mean(imgs, n=32):
    t = F.interpolate(torch.from_numpy(np.asarray(imgs, dtype=np.float32))[:, None], size=(n, n), mode="area")[:, 0]
    return t.mean(0)


def choose_rotation(slices, f_img):
    """Pick the 90-degree rotation that makes the mean MSD slice most similar to the mean figshare slice
    (anatomy only; no labels used)."""
    fm = thumb_mean(f_img[::7]); fm = (fm - fm.mean()) / (fm.std() + 1e-8)
    scores = []
    for k in range(4):
        r = np.rot90(slices, k, axes=(1, 2)).copy()
        mm = thumb_mean(r); mm = (mm - mm.mean()) / (mm.std() + 1e-8)
        scores.append(float((fm * mm).mean()))
    return int(np.argmax(scores)), scores


def load_msd(d, cache, per_patient, min_area, f_img, mres=128):
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return z["img"], z["mask"], z["pid"], int(z["rot"]), z["scores"].tolist()
    import nibabel as nib
    imgs_dir = glob.glob(os.path.join(d, "**", "imagesTr"), recursive=True)[0]
    files = sorted(f for f in glob.glob(os.path.join(imgs_dir, "*.nii.gz")) if not os.path.basename(f).startswith("._"))
    raw, masks, pids = [], [], []
    for i, f in enumerate(files):
        lp = f.replace("imagesTr", "labelsTr")
        if not os.path.exists(lp):
            continue
        vol = np.asarray(nib.load(f).dataobj[..., T1GD], dtype=np.float32)
        lab = np.asarray(nib.load(lp).dataobj)
        core = np.isin(lab, (2, 3))
        area = core.sum((0, 1))
        zs = [z for z in np.argsort(-area)[:per_patient] if area[z] >= min_area]
        for z in zs:
            raw.append(vol[:, :, z].copy()); masks.append(core[:, :, z].copy()); pids.append(os.path.basename(f).split(".")[0])
        if (i + 1) % 50 == 0:
            print(f"  read {i + 1}/{len(files)} volumes")
    raw = np.stack(raw); masks = np.stack(masks)
    rot, scores = choose_rotation(np.stack([(s - s.min()) / (s.max() - s.min() + 1e-8) for s in raw]), f_img)
    raw = np.rot90(raw, rot, axes=(1, 2)).copy(); masks = np.rot90(masks, rot, axes=(1, 2)).copy()
    out_i, out_m = [], []
    for s, m in zip(raw, masks):
        s = (s - s.min()) / (s.max() - s.min() + 1e-8)
        s = F.interpolate(torch.from_numpy(s)[None, None], size=(512, 512), mode="bilinear", align_corners=False)[0, 0]
        m = F.interpolate(torch.from_numpy(m.astype(np.float32))[None, None], size=(512, 512), mode="nearest")
        m = F.max_pool2d(m, 512 // mres)[0, 0]
        out_i.append(s.numpy().astype(np.float16)); out_m.append(m.numpy().astype(np.uint8))
    img, mask, pid = np.stack(out_i), np.stack(out_m), np.array(pids)
    np.savez(cache, img=img, mask=mask, pid=pid, rot=rot, scores=np.array(scores))
    return img, mask, pid, rot, scores


def fig_orientation(img, f_img, f_y, out):
    fig, ax = plt.subplots(1, 4, figsize=(9, 2.6), dpi=150)
    for k, j in enumerate([0, len(img) // 3, 2 * len(img) // 3]):
        ax[k].imshow(img[j].astype(np.float32), cmap="gray"); ax[k].set_title("MSD (oriented)", fontsize=8)
    jf = np.where(f_y == GLIOMA)[0][0]
    ax[3].imshow(f_img[jf].astype(np.float32), cmap="gray"); ax[3].set_title("figshare glioma", fontsize=8)
    for a in ax: a.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(out, "FigureX0_orientation_check.png")); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True, help="folder with <arch>_s<seed>.pt from run_external.py")
    ap.add_argument("--cache", required=True, help="figshare cache.npz")
    ap.add_argument("--out", default="results_msd")
    ap.add_argument("--msd_dir", default="/content/msd")
    ap.add_argument("--archs", nargs="+", default=["resnet50", "densenet121"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--per_patient", type=int, default=3, help="slices per patient (largest tumour-core area)")
    ap.add_argument("--min_area", type=int, default=100, help="minimum tumour-core pixels in a slice (240x240 grid)")
    ap.add_argument("--occlusion_archs", nargs="*", default=["resnet50"])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cam_res", type=int, default=128)
    ap.add_argument("--occ_stride", type=int, default=16, help="occlusion stride (keep 16 for the paper)")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "runs"), exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logf = open(os.path.join(args.out, "log.txt"), "a")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    f_img, _, f_y, _ = RE.load_dataset("unused", args.cache)
    mcache = os.path.join(args.out, "msd_cache.npz")
    if not os.path.exists(mcache):
        download_msd(args.msd_dir)
    img, mask, pid, rot, scores = load_msd(args.msd_dir, mcache, args.per_patient, args.min_area, f_img, args.cam_res)
    y = np.full(len(img), GLIOMA)
    log(f"MSD Task01: {len(img)} T1gd slices from {len(np.unique(pid))} glioma patients; rotation k={rot} "
        f"(similarity to figshare mean slice per k: {[round(s, 3) for s in scores]})")
    fig_orientation(img, f_img, f_y, args.out)
    idx = np.arange(len(img))
    ds = RE.SliceDS(img, y, idx, False)
    dl = DataLoader(ds, batch_size=args.batch, num_workers=args.workers)

    for arch in args.archs:
        for s in args.seeds:
            rp = os.path.join(args.out, "runs", f"{arch}_s{s}.npz")
            wp = os.path.join(args.models, f"{arch}_s{s}.pt")
            if os.path.exists(rp):
                log(f"[{arch} seed {s}] done already, skipping"); continue
            if not os.path.exists(wp):
                log(f"[{arch} seed {s}] weights not found at {wp}, skipping"); continue
            model, _ = RE.build_model(arch, False)
            model.load_state_dict(torch.load(wp, map_location="cpu")); model = model.to(dev)
            t0 = time.time()
            probs, _ = RM.predict_with_features(model, arch, dl, dev)
            cams = RM.multi_cam_maps(model, arch, ds, probs.argmax(1), dev, args.cam_res)
            save = {f"cam_{k}": v for k, v in cams.items()}
            if arch in args.occlusion_archs:
                save["cam_occlusion"] = RO.occlusion_maps(model, ds, probs.argmax(1), dev, args.cam_res, stride=args.occ_stride)
            acc = float((probs.argmax(1) == GLIOMA).mean())
            log(f"[{arch} seed {s}] glioma recall {acc:.4f}  ({time.time() - t0:.0f}s)")
            np.savez_compressed(rp, probs=probs.astype(np.float32), **save)
            del model; torch.cuda.empty_cache() if dev.type == "cuda" else None

    # ---- analysis
    L = ["EXTERNAL VALIDATION ON MSD Task01 (BraTS-derived gliomas) — SUMMARY",
         f"{len(img)} slices / {len(np.unique(pid))} patients; all gliomas; up to {args.per_patient} slices per patient; "
         "tumour core = labels 2+3; CIs from patient bootstrap.", ""]
    c = args.cam_res // 2
    L.append(f"Baselines (all slices): random point {100 * mask.reshape(len(mask), -1).mean():.1f}%, image centre {100 * mask[:, c, c].mean():.1f}%")
    for arch in args.archs:
        runs = [s for s in args.seeds if os.path.exists(os.path.join(args.out, "runs", f"{arch}_s{s}.npz"))]
        if len(runs) < 2:
            continue
        D = [np.load(os.path.join(args.out, "runs", f"{arch}_s{s}.npz")) for s in runs]
        pred = np.stack([d["probs"] for d in D]).argmax(2)
        rec = (pred == GLIOMA).mean(1)
        disagree = (pred != pred[0]).any(0); allc = (pred == GLIOMA).all(0)
        L.append(f"=== {arch} === glioma recall per seed: " + ", ".join(f"{100 * r:.1f}" for r in rec) +
                 f" (mean {100 * rec.mean():.1f} ± {100 * rec.std():.1f}); prediction disagreement {100 * disagree.mean():.1f}%; "
                 f"slices correct in every seed {int(allc.sum())} ({len(np.unique(pid[allc]))} patients)")
        if allc.sum() < 2:
            L.append("  too few slices correct in every seed"); continue
        methods = [m for m in ["gradcam", "gradcampp", "hirescam", "occlusion"] if f"cam_{m}" in D[0].files]
        per_method = {}
        L.append("  method | hit % | hit flips % | hit by all seeds % | median map r | peak distance mm | ensemble hit %")
        for m in methods:
            C = np.stack([d[f"cam_{m}"].astype(np.float32) for d in D])
            r = RM.multiplicity_block(C, mask, allc, args.cam_res)
            r["_slice"]["pid"] = pid[allc]
            per_method[m] = [r]
            L.append(f"  {m} | {100 * r['hit']:.1f} | {100 * r['hit_flip']:.1f} | {100 * r['all_hit']:.1f} | {r['pear']:.3f} | "
                     f"{r['peak_mm']:.1f} | {100 * r['ens_hit']:.1f}")
            if m in ("gradcam", "occlusion") and len(runs) >= 3:
                j = np.where(allc)[0][np.argmin(r["_pear_all"][allc])]
                RM.fig_seeds(img[j], C[:, j], r["_ens"][j], mask[j], runs, r["_hits"][:, j],
                             os.path.join(args.out, f"FigureX1_{arch}_{m}_seeds.png"))
        L += [x.replace("pooled over folds, ", "") for x in RM.ci_table(per_method, args)]
        L.append("")
    L.append("Peak distances use 0.49 mm/pixel at 512 px for comparability; MSD native spacing is 1 mm at 240 px "
             "(multiply by ~0.96 for native-scale mm).")
    open(os.path.join(args.out, "msd_summary.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    log(f"\nDone. Send {os.path.join(args.out, 'msd_summary.txt')} and FigureX*.png")


if __name__ == "__main__":
    main()
