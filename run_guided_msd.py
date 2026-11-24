#!/usr/bin/env python3
"""
Does mask-guided training also help on independent data?

Trains five mask-guided ResNet-50 models (seeds 0-4) on ALL figshare patients, with exactly the same 85/15 patient split
and settings as the unguided external models of run_external.py, and the same guidance loss as run_guided.py (lambda 1).
Then applies them to the MSD Task01 slices already prepared by run_external_msd.py (results_msd/msd_cache.npz) and
compares them with the unguided models (results_msd/runs/resnet50_s*.npz) on the same slices:
  - glioma recall and seed disagreement (all slices);
  - Grad-CAM and occlusion localisation on slices predicted as glioma by all seeds of BOTH conditions (paired,
    patient bootstrap).

Needs run_experiments.py, run_multiplicity.py, run_occlusion.py, run_guided.py, run_external_msd.py in the same folder.
Resumable: trained weights and MSD maps are saved per seed.

Usage:
  python run_guided_msd.py --cache results_mult2/cache.npz --msd results_msd --out results_msd_guided
"""
import argparse, json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_experiments as RE   # noqa: E402
import run_multiplicity as RM  # noqa: E402
import run_occlusion as RO     # noqa: E402
import run_guided as RG        # noqa: E402

GLIOMA = 0
PIX_MM = 0.49


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="figshare cache.npz")
    ap.add_argument("--msd", default="results_msd", help="output folder of run_external_msd.py")
    ap.add_argument("--out", default="results_msd_guided")
    ap.add_argument("--msd_dir", default="/content/msd", help="where to download MSD if the cache is missing")
    ap.add_argument("--data_dir", default="unused")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cam_res", type=int, default=128)
    ap.add_argument("--occ_stride", type=int, default=16, help="keep 16 (same as the unguided MSD run)")
    ap.add_argument("--no_pretrained", action="store_true", help="offline testing only")
    args = ap.parse_args()
    for d in ("models", "runs"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logf = open(os.path.join(args.out, "log.txt"), "a")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    log(f"Device: {dev} | lambda {args.lam}")

    # ---- data
    f_img, f_mask, f_y, f_pid = RE.load_dataset(args.data_dir, args.cache)
    mc = os.path.join(args.msd, "msd_cache.npz")
    if not os.path.exists(mc):
        log(f"{mc} not found: rebuilding the MSD slices (download ~7 GB + reading, about 30 min)")
        import run_external_msd as RX
        RX.download_msd(args.msd_dir)
        RX.load_msd(args.msd_dir, mc, 3, 100, f_img, args.cam_res)
    z = np.load(mc, allow_pickle=True)
    m_img, m_mask, m_pid = z["img"], z["mask"].astype(np.uint8), z["pid"]
    log(f"MSD rotation k = {int(z['rot'])} (must match the unguided run)")
    m_y = np.full(len(m_img), GLIOMA)
    log(f"MSD: {len(m_img)} slices / {len(np.unique(m_pid))} patients")
    ds_m = RE.SliceDS(m_img, m_y, np.arange(len(m_img)), False)
    dl_m = DataLoader(ds_m, batch_size=args.batch, num_workers=args.workers)

    # ---- same patient split as run_external.py
    pats = np.unique(f_pid); plab = np.array([np.bincount(f_y[f_pid == p]).argmax() for p in pats])
    tr_p, va_p = train_test_split(pats, test_size=0.15, stratify=plab, random_state=args.split_seed)
    tr = np.where(np.isin(f_pid, tr_p))[0]; va = np.where(np.isin(f_pid, va_p))[0]

    for s in args.seeds:
        rp = os.path.join(args.out, "runs", f"resnet50_s{s}.npz")
        if os.path.exists(rp):
            log(f"[guided seed {s}] done already, skipping"); continue
        wp = os.path.join(args.out, "models", f"resnet50_s{s}.pt")
        if os.path.exists(wp):
            model, _ = RE.build_model("resnet50", False)
            model.load_state_dict(torch.load(wp, map_location="cpu")); model = model.to(dev)
            log(f"\n=== guided resnet50 | seed {s} | loaded saved weights ===")
        else:
            log(f"\n=== guided resnet50 | seed {s} | train {len(tr)} val {len(va)} figshare slices ===")
            t0 = time.time()
            model, n_ep, _ = RG.train_guided(tr, va, f_img, f_mask, f_y, args, dev, log, s)
            for h in list(model.layer4._forward_hooks):
                model.layer4._forward_hooks.pop(h)
            torch.save(model.state_dict(), wp)
            log(f"    trained in {time.time() - t0:.0f}s ({n_ep} epochs)")
        t0 = time.time()
        probs, _ = RM.predict_with_features(model, "resnet50", dl_m, dev)
        pred = probs.argmax(1)
        g = RM.multi_cam_maps(model, "resnet50", ds_m, pred, dev, args.cam_res, methods=["gradcam"])["gradcam"]
        o = RO.occlusion_maps(model, ds_m, pred, dev, args.cam_res, stride=args.occ_stride)
        rec = float((pred == GLIOMA).mean())
        log(f"[guided seed {s}] MSD glioma recall {rec:.4f}  ({time.time() - t0:.0f}s)")
        np.savez_compressed(rp, probs=probs.astype(np.float32), cam_gradcam=g, cam_occlusion=o)
        del model; torch.cuda.empty_cache() if dev.type == "cuda" else None

    analyse(args, m_img, m_mask, m_pid, log)
    log(f"\nDone. Send {args.out}/guided_msd_summary.txt and {args.out}/FigureX2_guided_msd.png")


def analyse(args, m_img, mask, pid, log):
    S = args.seeds
    G = [np.load(os.path.join(args.out, "runs", f"resnet50_s{s}.npz")) for s in S]
    U = [np.load(os.path.join(args.msd, "runs", f"resnet50_s{s}.npz")) for s in S]
    pg = np.stack([d["probs"].argmax(1) for d in G]); pu = np.stack([d["probs"].argmax(1) for d in U])
    rg = (pg == GLIOMA).mean(1) * 100; ru = (pu == GLIOMA).mean(1) * 100
    dis = lambda p: 100 * (p != p[0]).any(0).mean()
    L = ["MASK-GUIDED vs UNGUIDED ResNet-50 on independent MSD Task01 gliomas (models trained on all figshare patients)",
         f"lambda = {args.lam}; {len(pid)} slices / {len(np.unique(pid))} patients; 95% CIs from 2,000 patient bootstrap resamples", ""]
    try:
        pw = wilcoxon(rg, ru).pvalue
    except ValueError:
        pw = float("nan")
    L.append("glioma recall per seed: guided " + ", ".join(f"{v:.1f}" for v in rg) + f" (mean {rg.mean():.1f} ± {rg.std(ddof=1):.1f}) | "
             "unguided " + ", ".join(f"{v:.1f}" for v in ru) + f" (mean {ru.mean():.1f} ± {ru.std(ddof=1):.1f})")
    L.append(f"  paired difference in mean recall {rg.mean() - ru.mean():+.1f} pp (Wilcoxon over seeds P = {pw:.3f}, n = {len(S)}; low power)")
    # slice-level recall difference with patient bootstrap (recall averaged over seeds per slice)
    r = dict(pid=pid, rg=(pg == GLIOMA).mean(0), ru=(pu == GLIOMA).mean(0))
    r["d"] = r["rg"] - r["ru"]
    ci = RM.patient_bootstrap(r, ["rg", "ru", "d"]); p_d, _ = RM.boot_p(r, "d")
    f = lambda t, mul=100: f"{mul * t[0]:.1f} ({mul * t[1]:.1f} to {mul * t[2]:.1f})"
    fp = lambda p: "<0.0005" if p < 0.0005 else f"{p:.4f}"
    L.append(f"  mean recall over seeds (patient bootstrap): guided {f(ci['rg'])} | unguided {f(ci['ru'])} | diff {f(ci['d'])} pp [P {fp(p_d)}]")
    L.append(f"  seed disagreement on predicted class: guided {dis(pg):.1f}% | unguided {dis(pu):.1f}%")
    L.append(f"  slices predicted glioma by all seeds: guided {int((pg == GLIOMA).all(0).sum())} | unguided {int((pu == GLIOMA).all(0).sum())}")

    both = (pg == GLIOMA).all(0) & (pu == GLIOMA).all(0)
    L.append(f"\nLocalisation on slices predicted glioma by all seeds of BOTH conditions: {int(both.sum())} slices / "
             f"{len(np.unique(pid[both]))} patients")
    res = args.cam_res; c = res // 2
    ex = {}
    if both.sum() >= 5:
        for m in ("gradcam", "occlusion"):
            Cg = np.stack([d[f"cam_{m}"].astype(np.float32) for d in G]); Cu = np.stack([d[f"cam_{m}"].astype(np.float32) for d in U])
            bg = RM.multiplicity_block(Cg, mask, both, res); bu = RM.multiplicity_block(Cu, mask, both, res)
            a, b = bg["_slice"], bu["_slice"]
            rr = dict(pid=pid[both])
            for k in ("hit", "flip", "all_hit", "ens_hit"):
                rr[f"g_{k}"] = a[k]; rr[f"u_{k}"] = b[k]; rr[f"d_{k}"] = a[k] - b[k]
            rr["centre"] = a["centre"]; rr["g_dc"] = a["hit"] - a["centre"]
            keys = [k for k in rr if k != "pid"]
            ci = RM.patient_bootstrap(rr, keys)
            L.append(f"--- {m} ---")
            for k, name in (("hit", "hit rate %"), ("flip", "hit flips %"), ("all_hit", "hit by all seeds %"), ("ens_hit", "ensemble hit %")):
                p, _ = RM.boot_p(rr, f"d_{k}")
                L.append(f"  {name:20s} guided {f(ci['g_' + k])} | unguided {f(ci['u_' + k])} | diff {f(ci['d_' + k])} pp [P {fp(p)}]")
            p, _ = RM.boot_p(rr, "g_dc")
            L.append(f"  map r between seeds: guided {bg['pear']:.3f} | unguided {bu['pear']:.3f};  "
                     f"peak distance mm: guided {bg['peak_mm']:.1f} | unguided {bu['peak_mm']:.1f}")
            L.append(f"  guided hit minus image centre {f(ci['g_dc'])} pp [P {fp(p)}] (centre {f(ci['centre'])} %)")
            ex[m] = (bg["_ens"], bu["_ens"])
        # figure: 4 randomly chosen (not cherry-picked) slices, ensemble Grad-CAM unguided vs guided
        rng = np.random.default_rng(2026)
        js = rng.choice(np.where(both)[0], size=min(4, int(both.sum())), replace=False)
        eg, eu = ex["gradcam"]
        fig, ax = plt.subplots(len(js), 3, figsize=(6.6, 2.25 * len(js)), dpi=200)
        ax = np.atleast_2d(ax)
        up = lambda a: F.interpolate(torch.from_numpy(a)[None, None], size=(512, 512), mode="bilinear", align_corners=False)[0, 0].numpy()
        for r_, j in enumerate(js):
            im = m_img[j].astype(np.float32)
            mk = F.interpolate(torch.from_numpy(mask[j].astype(np.float32))[None, None], size=(512, 512), mode="nearest")[0, 0].numpy()
            ax[r_, 0].imshow(im, cmap="gray"); ax[r_, 0].contour(mk, levels=[0.5], colors="yellow", linewidths=0.8)
            for k, (E, t) in enumerate(((eu, "Unguided (mean of 5 seeds)"), (eg, "Mask-guided (mean of 5 seeds)"))):
                ax[r_, k + 1].imshow(im, cmap="gray"); ax[r_, k + 1].imshow(up(E[j]), cmap="jet", alpha=0.45)
                ax[r_, k + 1].contour(mk, levels=[0.5], colors="white", linewidths=0.8)
                if r_ == 0: ax[r_, k + 1].set_title(t, fontsize=7)
            if r_ == 0: ax[r_, 0].set_title("MSD slice + tumour core", fontsize=7)
            for a_ in ax[r_]: a_.axis("off")
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "FigureX2_guided_msd.png")); plt.close(fig)
    else:
        L.append("  too few slices for a localisation comparison")
    open(os.path.join(args.out, "guided_msd_summary.txt"), "w").write("\n".join(L) + "\n")
    log("\n".join(L))


if __name__ == "__main__":
    main()
