#!/usr/bin/env python3
"""
Occlusion sensitivity (a perturbation-based, gradient-free explanation) for the explanation-multiplicity study.

Retrains the same models as run_multiplicity.py (same folds, same seeds) and, for every test slice, computes
  - an occlusion map: drop in the predicted-class probability when a 32x32 patch (stride 16, on the 224x224 input)
    is replaced by black background, averaged over overlapping patches;
  - the Grad-CAM map of the same model,
then reports seed dependence, baselines and Grad-CAM-vs-occlusion agreement, as in run_multiplicity.py.

Needs run_experiments.py and run_multiplicity.py in the same folder. Resumable.

Usage:
  python run_occlusion.py --out results_occ --cache results_mult2/cache.npz              # ResNet-50 (default)
  python run_occlusion.py --out results_occ --cache results_mult2/cache.npz --archs resnet50 densenet121
"""
import argparse, json, os, sys, time
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_experiments as RE   # noqa: E402
import run_multiplicity as RM  # noqa: E402

METHODS = ["occlusion", "gradcam"]


@torch.no_grad()
def occlusion_maps(model, ds, pred, dev, res, patch=32, stride=16, bs=176):
    model.eval()
    fill = ((0 - RE.MEAN) / RE.STD).view(1, 3, 1, 1).to(dev)              # black background after normalisation
    pos = [(y, x) for y in range(0, 224 - patch + 1, stride) for x in range(0, 224 - patch + 1, stride)]
    cover = torch.zeros(224, 224, device=dev)
    for y, x in pos:
        cover[y:y + patch, x:x + patch] += 1
    out = []
    for i in range(len(ds)):
        x0 = ds[i][0].to(dev)[None]
        c = int(pred[i])
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            p0 = torch.softmax(model(x0).float(), 1)[0, c]
        drops = []
        for s in range(0, len(pos), bs):
            chunk = pos[s:s + bs]
            xb = x0.repeat(len(chunk), 1, 1, 1)
            for k, (y, x) in enumerate(chunk):
                xb[k, :, y:y + patch, x:x + patch] = fill[0]
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                drops.append(p0 - torch.softmax(model(xb).float(), 1)[:, c])
        drops = torch.cat(drops)
        m = torch.zeros(224, 224, device=dev)
        for d, (y, x) in zip(drops, pos):
            m[y:y + patch, x:x + patch] += d
        m = F.relu(m / cover.clamp(min=1))
        m = F.interpolate(m[None, None], size=(res, res), mode="area")[0, 0]
        m = m / (m.max() + 1e-8)
        out.append(m.cpu().numpy().astype(np.float16))
    return np.stack(out)


def run_path(args, arch, f, s):
    return os.path.join(args.out, "runs", f"{arch}_f{f}_s{s}.npz")


def analyse(args, y, pid, mask_lr, fold, img):
    res = args.cam_res
    L = ["OCCLUSION vs GRAD-CAM — EXPLANATION MULTIPLICITY SUMMARY",
         f"folds {args.folds}, seeds {args.seeds}; occlusion patch 32 px, stride 16 px on 224 px input; maps analysed at {res}x{res}", ""]
    centre_all = []
    for arch in args.archs:
        per_method = {m: [] for m in METHODS}
        accs, fold_rows, cross_r, cross_d, cross_hit = [], [], [], [], []
        fig_done = False
        for f in range(args.folds):
            runs = [s for s in args.seeds if os.path.exists(run_path(args, arch, f, s))]
            if len(runs) < 2:
                continue
            D = [np.load(run_path(args, arch, f, s)) for s in runs]
            te = D[0]["te"]; yt = y[te]; masks = mask_lr[te]
            pred = np.stack([d["probs"] for d in D]).argmax(2)
            accs += list((pred == yt).mean(1))
            allc = (pred == yt).all(0)
            if arch == args.archs[0]:
                centre_all.append(masks[:, res // 2, res // 2].mean())
            if allc.sum() < 2:
                continue
            row = f"  fold {f}: acc per seed " + ", ".join(f"{100 * a:.1f}" for a in (pred == yt).mean(1))
            Cs = {}
            for m in METHODS:
                C = np.stack([d[f"cam_{m}"].astype(np.float32) for d in D])
                Cs[m] = C
                r = RM.multiplicity_block(C, masks, allc, res)
                r["_slice"]["pid"] = pid[te][allc]
                per_method[m].append(r)
                row += f" | {m}: hit {100 * r['hit']:.1f}% flips {100 * r['hit_flip']:.1f}% r {r['pear']:.2f} peak {r['peak_mm']:.1f} mm"
                if m == "occlusion" and not fig_done and len(runs) >= 3:
                    j = np.where(allc)[0][np.argmin(r["_pear_all"][allc])]
                    RM.fig_seeds(img[te[j]], C[:, j], r["_ens"][j], masks[j], runs, r["_hits"][:, j],
                                 os.path.join(args.out, f"FigureO1_occlusion_seeds_{arch}.png"))
                    fig_done = True
            # agreement between methods within the same model
            o, g = Cs["occlusion"][:, allc], Cs["gradcam"][:, allc]
            n = allc.sum()
            cross_r.append(np.median(np.mean([RM.pearson_rows(o[s].reshape(n, -1), g[s].reshape(n, -1)) for s in range(len(runs))], 0)))
            cross_d.append(np.median(np.mean([np.linalg.norm(RM.peak_xy(o[s]) - RM.peak_xy(g[s]), axis=1) for s in range(len(runs))], 0)) * (512 / res * RM.PIX_MM))
            ho = np.stack([RM.loc_metrics(o[s], masks[allc])[0] for s in range(len(runs))])
            hg = np.stack([RM.loc_metrics(g[s], masks[allc])[0] for s in range(len(runs))])
            cross_hit.append((ho == hg).mean())
            fold_rows.append(row)
        if not accs:
            continue
        L.append(f"=== {arch} === accuracy over runs {100 * np.mean(accs):.1f} ± {100 * np.std(accs):.1f} % (n = {len(accs)})")
        L += fold_rows
        L.append("TABLE O1 — seed dependence by method (fold means ± SD; slices correct in every seed)")
        L.append("  method | hit rate % | hit flips % | hit by all seeds % | median map r | top-20% IoU | peak distance mm | ensemble hit %")
        for m in METHODS:
            rs = per_method[m]
            if not rs:
                continue
            g_ = lambda k: [x[k] for x in rs]
            L.append(f"  {m} | {100 * np.mean(g_('hit')):.1f} ± {100 * np.std(g_('hit')):.1f} | {100 * np.mean(g_('hit_flip')):.1f} ± {100 * np.std(g_('hit_flip')):.1f} | "
                     f"{100 * np.mean(g_('all_hit')):.1f} ± {100 * np.std(g_('all_hit')):.1f} | {np.mean(g_('pear')):.3f} ± {np.std(g_('pear')):.3f} | "
                     f"{np.mean(g_('tiou')):.3f} | {np.mean(g_('peak_mm')):.1f} ± {np.std(g_('peak_mm')):.1f} | {100 * np.mean(g_('ens_hit')):.1f} ± {100 * np.std(g_('ens_hit')):.1f}")
        L += RM.ci_table(per_method, args)
        if cross_r:
            L.append(f"  Occlusion vs Grad-CAM of the SAME model: median map r {np.mean(cross_r):.3f} ± {np.std(cross_r):.3f}; "
                     f"peak distance {np.mean(cross_d):.1f} ± {np.std(cross_d):.1f} mm; same hit/miss outcome {100 * np.mean(cross_hit):.1f} ± {100 * np.std(cross_hit):.1f} %")
        L.append("")
    if centre_all:
        L.append(f"Image-centre baseline (all test slices): {100 * np.mean(centre_all):.1f} %")
    open(os.path.join(args.out, "occlusion_summary.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results_occ")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--data_dir", default="data/figshare_brain_tumour")
    ap.add_argument("--archs", nargs="+", default=["resnet50"], choices=["resnet50", "densenet121"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cam_res", type=int, default=128)
    ap.add_argument("--analyse_only", action="store_true")
    ap.add_argument("--no_pretrained", action="store_true", help="offline testing only")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "runs"), exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logf = open(os.path.join(args.out, "log.txt"), "a")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    log(f"Device: {dev} | torch {torch.__version__}")
    img, mask, y, pid = RE.load_dataset(args.data_dir, args.cache)
    mask_lr = F.max_pool2d(torch.from_numpy(mask.astype(np.float32))[:, None], kernel_size=512 // args.cam_res)[:, 0].numpy().astype(np.uint8)
    fold = RM.make_folds(y, pid, args.folds, args.split_seed)
    if not args.analyse_only:
        for f in range(args.folds):
            tr, va, te = RM.fold_split(y, pid, fold, f, args.split_seed)
            ds_te = RE.SliceDS(img, y, te, False)
            dl_te = DataLoader(ds_te, batch_size=args.batch, num_workers=args.workers)
            for arch in args.archs:
                for s in args.seeds:
                    p = run_path(args, arch, f, s)
                    if os.path.exists(p):
                        log(f"[{arch} fold {f} seed {s}] done already, skipping"); continue
                    log(f"\n=== {arch} | fold {f} | seed {s} | train {len(tr)} val {len(va)} test {len(te)} ===")
                    t0 = time.time()
                    model, n_ep, best = RM.train_one(arch, tr, va, img, y, args, dev, log, s)
                    probs, _ = RM.predict_with_features(model, arch, dl_te, dev)
                    acc = float((probs.argmax(1) == y[te]).mean())
                    log(f"    test acc {acc:.4f}  ({time.time() - t0:.0f}s); computing Grad-CAM and occlusion ...")
                    g = RM.multi_cam_maps(model, arch, ds_te, probs.argmax(1), dev, args.cam_res, methods=["gradcam"])["gradcam"]
                    t1 = time.time()
                    o = occlusion_maps(model, ds_te, probs.argmax(1), dev, args.cam_res)
                    log(f"    occlusion done in {time.time() - t1:.0f}s")
                    np.savez_compressed(p, te=te, probs=probs.astype(np.float32), cam_gradcam=g, cam_occlusion=o)
                    json.dump(dict(accuracy=acc, epochs=n_ep), open(p.replace(".npz", ".json"), "w"))
                    del model; torch.cuda.empty_cache() if dev.type == "cuda" else None
    analyse(args, y, pid, mask_lr, fold, img)
    log(f"\nDone. Send {os.path.join(args.out, 'occlusion_summary.txt')} and FigureO1_*.png")


if __name__ == "__main__":
    main()
