#!/usr/bin/env python3
"""
Remedy experiment: mask-guided training ("right for the right reasons").

ResNet-50 is trained exactly as in run_occlusion.py (same patient-wise folds, same seeds, same settings) but with an
auxiliary loss that penalises class-activation energy outside the expert tumour mask:

    loss = cross-entropy + lambda * sum(CAM_y * (1 - M)) / sum(CAM_y)

where CAM_y = ReLU(sum_k w_{y,k} A_k) is the class activation map of the TRUE class from the last residual stage
(for ResNet-50 with global average pooling + linear head this is proportional to Grad-CAM) and M is the tumour
mask max-pooled to the 7 x 7 feature grid (any cell touching the tumour counts as inside). lambda is fixed a priori
(default 1.0); early stopping uses the validation cross-entropy, as for the unguided models.

Afterwards Grad-CAM and occlusion maps are computed for every test slice, and the guided models are compared with
the unguided ones (results of run_occlusion.py) slice by slice: accuracy, localisation, seed dependence.

Needs run_experiments.py, run_multiplicity.py, run_occlusion.py in the same folder. Resumable.

Usage:
  python run_guided.py --cache results_mult2/cache.npz --out results_guided --baseline results_occ
"""
import argparse, json, os, random, sys, time
from itertools import combinations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_experiments as RE   # noqa: E402
import run_multiplicity as RM  # noqa: E402
import run_occlusion as RO     # noqa: E402

CLASSES = RE.CLASSES


class GuidedDS(Dataset):
    """Like RE.SliceDS, but also returns the tumour mask (224 x 224), augmented jointly with the image."""
    def __init__(self, img, mask, y, idx, train):
        self.img, self.mask, self.y, self.idx = img, mask, y, idx
        self.geo = T.Compose([T.RandomAffine(degrees=10, translate=(0.1, 0.1)), T.RandomHorizontalFlip()]) if train else None
        self.col = T.ColorJitter(brightness=0.1, contrast=0.1) if train else None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x = torch.from_numpy(self.img[j].astype(np.float32))[None, None]
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)[0].repeat(3, 1, 1)
        m = torch.from_numpy(self.mask[j].astype(np.float32))[None, None]
        m = F.interpolate(m, size=(224, 224), mode="nearest")[0]
        if self.geo is not None:
            x, m = self.geo(tv_tensors.Image(x), tv_tensors.Mask(m))
            x = self.col(x).clamp(0, 1)
        x = (torch.as_tensor(x) - RE.MEAN) / RE.STD
        return x, int(self.y[j]), torch.as_tensor(m)[0]


def cam_outside(model, A, y, m224):
    """Fraction of true-class CAM energy outside the (7x7 max-pooled) mask."""
    W = model.fc[1].weight                                  # (classes, 2048)
    cam = F.relu(torch.einsum("nkhw,nk->nhw", A, W[y]))
    m7 = F.adaptive_max_pool2d(m224[:, None], A.shape[-2:])[:, 0]
    has = m7.flatten(1).amax(1) > 0
    frac = (cam * (1 - m7)).flatten(1).sum(1) / (cam.flatten(1).sum(1) + 1e-6)
    return (frac * has).sum() / has.sum().clamp(min=1)


def train_guided(tr, va, img, mask, y, args, dev, log, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model, _ = RE.build_model("resnet50", not args.no_pretrained)
    model = model.to(dev)
    store = {}
    model.layer4.register_forward_hook(lambda m, i, o: store.__setitem__("A", o))
    cw = torch.tensor(len(tr) / (len(CLASSES) * np.bincount(y[tr], minlength=len(CLASSES))), dtype=torch.float32).to(dev)
    crit = nn.CrossEntropyLoss(weight=cw)
    dl_tr = DataLoader(GuidedDS(img, mask, y, tr, True), batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       pin_memory=dev.type == "cuda")
    dl_va = DataLoader(GuidedDS(img, mask, y, va, False), batch_size=args.batch, num_workers=args.workers)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=5)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    best, best_state, bad = 1e9, None, 0
    for ep in range(args.epochs):
        model.train(); tl = to = 0; n = 0
        for x, t, m in dl_tr:
            x, t, m = x.to(dev), t.to(dev), m.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                ce = crit(model(x), t)
            out = cam_outside(model, store["A"].float(), t, m)
            loss = ce + args.lam * out
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += ce.item() * len(t); to += out.item() * len(t); n += len(t)
        model.eval(); vl = vo = 0; nv = 0
        with torch.no_grad():
            for x, t, m in dl_va:
                x, t, m = x.to(dev), t.to(dev), m.to(dev)
                with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                    vl += crit(model(x).float(), t).item() * len(t)
                vo += cam_outside(model, store["A"].float(), t, m).item() * len(t); nv += len(t)
        vl /= nv; sched.step(vl)
        log(f"    epoch {ep + 1:3d}  train CE {tl / n:.4f}  train outside {to / n:.3f}  val CE {vl:.4f}  val outside {vo / nv:.3f}")
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                log(f"    early stop at epoch {ep + 1}"); break
    model.load_state_dict(best_state)
    return model, ep + 1, best


def run_path(out, f, s):
    return os.path.join(out, "runs", f"resnet50_f{f}_s{s}.npz")


def compare(args, y, pid, mask_lr, fold, img):
    """Guided vs unguided, slice by slice on slices correct in all seeds of BOTH conditions."""
    res = args.cam_res
    L = ["MASK-GUIDED vs UNGUIDED ResNet-50 — paired comparison (same folds, same seeds, same test slices)",
         f"lambda = {args.lam}; slices correct in every seed of both conditions; 95% CIs from 2,000 patient bootstrap resamples", ""]
    acc_g, acc_u = [], []
    recs = {m: [] for m in ("gradcam", "occlusion")}
    example = None
    for f in range(args.folds):
        runs = [s for s in args.seeds if os.path.exists(run_path(args.out, f, s)) and os.path.exists(run_path(args.baseline, f, s))]
        if len(runs) < 2:
            continue
        G = [np.load(run_path(args.out, f, s)) for s in runs]; U = [np.load(run_path(args.baseline, f, s)) for s in runs]
        te = G[0]["te"]; assert np.array_equal(te, U[0]["te"]), "test slices differ between conditions"
        yt = y[te]; masks = mask_lr[te]
        pg = np.stack([d["probs"] for d in G]).argmax(2); pu = np.stack([d["probs"] for d in U]).argmax(2)
        acc_g += list((pg == yt).mean(1)); acc_u += list((pu == yt).mean(1))
        both = (pg == yt).all(0) & (pu == yt).all(0)
        if both.sum() < 2:
            continue
        for m in recs:
            Cg = np.stack([d[f"cam_{m}"].astype(np.float32) for d in G]); Cu = np.stack([d[f"cam_{m}"].astype(np.float32) for d in U])
            rg = RM.multiplicity_block(Cg, masks, both, res); ru = RM.multiplicity_block(Cu, masks, both, res)
            d = dict(pid=pid[te][both])
            for k in ("hit", "flip", "all_hit", "ens_hit", "centre"):
                d[f"g_{k}"] = rg["_slice"][k]; d[f"u_{k}"] = ru["_slice"][k]; d[f"d_{k}"] = rg["_slice"][k] - ru["_slice"][k]
            d["g_r"] = rg["_pear_all"][both]; d["u_r"] = ru["_pear_all"][both]; d["d_r"] = d["g_r"] - d["u_r"]
            d["g_dc"] = d["g_hit"] - d["g_centre"]
            recs[m].append(d)
            if m == "gradcam" and example is None and len(runs) >= 3:
                j = np.where(both)[0][np.argmax(d["d_hit"] + 0.5 * d["d_r"])]
                example = (img[te[j]], Cu[:, j], Cg[:, j], masks[j], runs)
    acc_g, acc_u = np.array(acc_g), np.array(acc_u)
    if len(acc_g):
        p = wilcoxon(acc_g, acc_u).pvalue if np.any(acc_g != acc_u) else 1.0
        L.append(f"Accuracy over fold-seed runs: guided {100 * acc_g.mean():.1f} ± {100 * acc_g.std():.1f} %, unguided "
                 f"{100 * acc_u.mean():.1f} ± {100 * acc_u.std():.1f} %; paired difference {100 * (acc_g - acc_u).mean():+.1f} pp "
                 f"(Wilcoxon P = {p:.3g}, n = {len(acc_g)})")
    f = lambda t, mul=100: f"{mul * t[0]:.1f} ({mul * t[1]:.1f} to {mul * t[2]:.1f})"
    for m, rs in recs.items():
        if not rs:
            continue
        R = {k: np.concatenate([r[k] for r in rs]) for k in rs[0]}
        keys = ["g_hit", "u_hit", "d_hit", "g_flip", "u_flip", "d_flip", "g_all_hit", "u_all_hit", "d_all_hit",
                "g_ens_hit", "u_ens_hit", "g_r", "u_r", "d_r", "g_centre", "g_dc"]
        ci = RM.patient_bootstrap(R, keys)
        P = {k: RM.boot_p(R, k)[0] for k in ("d_hit", "d_flip", "d_all_hit", "d_r", "g_dc")}
        fp = lambda p: "<0.0005" if p == 0 else f"{p:.4f}"
        L += [f"--- {m} (n = {len(R['pid'])} slices, {len(np.unique(R['pid']))} patients) ---",
              f"  hit rate %:            guided {f(ci['g_hit'])} | unguided {f(ci['u_hit'])} | diff {f(ci['d_hit'])} pp [P {fp(P['d_hit'])}]",
              f"  hit flips %:           guided {f(ci['g_flip'])} | unguided {f(ci['u_flip'])} | diff {f(ci['d_flip'])} pp [P {fp(P['d_flip'])}]",
              f"  hit by all seeds %:    guided {f(ci['g_all_hit'])} | unguided {f(ci['u_all_hit'])} | diff {f(ci['d_all_hit'])} pp [P {fp(P['d_all_hit'])}]",
              f"  map r between seeds:   guided {f(ci['g_r'], 1)} | unguided {f(ci['u_r'], 1)} | diff {f(ci['d_r'], 1)} [P {fp(P['d_r'])}]",
              f"  ensemble hit %:        guided {f(ci['g_ens_hit'])} | unguided {f(ci['u_ens_hit'])}",
              f"  guided hit minus image centre: {f(ci['g_dc'])} pp [P {fp(P['g_dc'])}] (centre {f(ci['g_centre'])} %)", ""]
    if example is not None:
        im, Cu, Cg, mk, runs = example
        n = len(runs)
        fig, ax = plt.subplots(2, n + 1, figsize=(2.2 * (n + 1), 4.6), dpi=200)
        im = im.astype(np.float32)
        up = lambda c: F.interpolate(torch.from_numpy(np.asarray(c, np.float32))[None, None], size=im.shape, mode="bilinear")[0, 0].numpy()
        mkk = F.interpolate(torch.from_numpy(mk.astype(np.float32))[None, None], size=im.shape, mode="nearest")[0, 0].numpy()
        for r, (C, lab) in enumerate([(Cu, "Unguided"), (Cg, "Mask-guided")]):
            ax[r, 0].imshow(im, cmap="gray"); ax[r, 0].contour(mkk, levels=[0.5], colors="yellow", linewidths=0.8)
            ax[r, 0].set_title(lab, fontsize=8)
            for i in range(n):
                ax[r, i + 1].imshow(im, cmap="gray"); ax[r, i + 1].imshow(up(C[i]), cmap="jet", alpha=0.45)
                ax[r, i + 1].contour(mkk, levels=[0.5], colors="white", linewidths=0.8); ax[r, i + 1].set_title(f"Seed {runs[i]}", fontsize=8)
            for a in ax[r]: a.axis("off")
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "FigureG1_guided_vs_unguided.png")); plt.close(fig)
    open(os.path.join(args.out, "guided_vs_unguided_summary.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", default="results_guided")
    ap.add_argument("--baseline", default="results_occ", help="output folder of run_occlusion.py (unguided models)")
    ap.add_argument("--data_dir", default="data/figshare_brain_tumour")
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cam_res", type=int, default=128)
    ap.add_argument("--occ_stride", type=int, default=16, help="occlusion stride (keep 16 for the paper)")
    ap.add_argument("--analyse_only", action="store_true")
    ap.add_argument("--no_pretrained", action="store_true", help="offline testing only")
    args = ap.parse_args()
    args.archs = ["resnet50"]
    os.makedirs(os.path.join(args.out, "runs"), exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logf = open(os.path.join(args.out, "log.txt"), "a")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    log(f"Device: {dev} | lambda {args.lam}")
    img, mask, y, pid = RE.load_dataset(args.data_dir, args.cache)
    mask_lr = F.max_pool2d(torch.from_numpy(mask.astype(np.float32))[:, None], kernel_size=512 // args.cam_res)[:, 0].numpy().astype(np.uint8)
    fold = RM.make_folds(y, pid, args.folds, args.split_seed)
    if not args.analyse_only:
        for f in range(args.folds):
            tr, va, te = RM.fold_split(y, pid, fold, f, args.split_seed)
            ds_te = RE.SliceDS(img, y, te, False)
            dl_te = DataLoader(ds_te, batch_size=args.batch, num_workers=args.workers)
            for s in args.seeds:
                p = run_path(args.out, f, s)
                if os.path.exists(p):
                    log(f"[guided fold {f} seed {s}] done already, skipping"); continue
                log(f"\n=== guided resnet50 | fold {f} | seed {s} ===")
                t0 = time.time()
                model, n_ep, best = train_guided(tr, va, img, mask, y, args, dev, log, s)
                for h in list(model.layer4._forward_hooks):
                    model.layer4._forward_hooks.pop(h)
                probs, _ = RM.predict_with_features(model, "resnet50", dl_te, dev)
                acc = float((probs.argmax(1) == y[te]).mean())
                g = RM.multi_cam_maps(model, "resnet50", ds_te, probs.argmax(1), dev, args.cam_res, methods=["gradcam"])["gradcam"]
                o = RO.occlusion_maps(model, ds_te, probs.argmax(1), dev, args.cam_res, stride=args.occ_stride)
                log(f"    test acc {acc:.4f}  ({time.time() - t0:.0f}s incl. maps)")
                np.savez_compressed(p, te=te, probs=probs.astype(np.float32), cam_gradcam=g, cam_occlusion=o)
                json.dump(dict(accuracy=acc, epochs=n_ep, val_ce=best), open(p.replace(".npz", ".json"), "w"))
                del model; torch.cuda.empty_cache() if dev.type == "cuda" else None
    RO.analyse(args, y, pid, mask_lr, fold, img)
    if os.path.exists(os.path.join(args.out, "occlusion_summary.txt")):
        os.replace(os.path.join(args.out, "occlusion_summary.txt"), os.path.join(args.out, "guided_multiplicity_summary.txt"))
    if os.path.isdir(os.path.join(args.baseline, "runs")):
        compare(args, y, pid, mask_lr, fold, img)
    else:
        log(f"Baseline folder {args.baseline} not found: skipping the guided-vs-unguided comparison")
    log(f"\nDone. Send {args.out}/guided_vs_unguided_summary.txt, guided_multiplicity_summary.txt and FigureG1/FigureO1 PNGs")


if __name__ == "__main__":
    main()
