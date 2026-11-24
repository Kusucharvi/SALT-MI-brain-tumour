#!/usr/bin/env python3
"""
Explanation multiplicity and leakage fingerprints in brain tumour MRI classification.

Idea 1 - Explanation multiplicity: the same model is retrained with several random seeds on each of
          5 patient-wise folds. For every test slice we compare the seeds' Grad-CAM maps with each other
          and with the expert tumour mask (agreement, hit flips, ensemble map).
Idea 2 - Leakage fingerprint: per fold, a model trained with a leaky slice-level split is compared with
          the clean (patient-split) model on how well its features re-identify the PATIENT of a slice.

Needs run_experiments.py in the same folder (data loading, models, training, Grad-CAM are reused).

Usage:
  python run_multiplicity.py --out results_mult                       # ResNet-50, 5 folds x 5 seeds + leaky runs
  python run_multiplicity.py --out results_mult --archs resnet50 densenet121
  python run_multiplicity.py --out results_mult --cache results_v2/cache.npz   # reuse an existing data cache
Re-running the same command resumes; finished runs are skipped.
"""
import argparse, glob, json, os, random, sys, time
from itertools import combinations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from scipy.stats import mannwhitneyu, spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_experiments as RE  # noqa: E402

CLASSES = RE.CLASSES
PIX_MM = 0.49  # in-plane pixel size of the dataset at 512 x 512 (mm)


# ------------------------------------------------------------------ helpers
def target_layer(model, arch):
    return model.layer4 if arch == "resnet50" else model.features


def feature_hook(model, arch, store):
    if arch == "resnet50":
        return model.avgpool.register_forward_hook(lambda m, i, o: store.append(o.flatten(1).detach().float().cpu()))
    return model.classifier.register_forward_pre_hook(lambda m, i: store.append(i[0].detach().float().cpu()))


@torch.no_grad()
def predict_with_features(model, arch, dl, dev):
    model.eval(); store = []; P = []
    h = feature_hook(model, arch, store)
    for x, _ in dl:
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            P.append(torch.softmax(model(x.to(dev)).float(), 1).cpu().numpy())
    h.remove()
    return np.concatenate(P), torch.cat(store).numpy().astype(np.float16)


def gradcam_maps(model, arch, ds, pred, dev, res, bs=16):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    gc = RE.GradCAM(model, target_layer(model, arch))
    out = []
    for s in range(0, len(ds), bs):
        ii = list(range(s, min(s + bs, len(ds))))
        x = torch.stack([ds[i][0] for i in ii]).to(dev)
        cam = gc(x, torch.tensor(pred[ii], device=dev))            # (b, 512, 512), max-normalised
        cam = F.interpolate(torch.from_numpy(cam)[:, None], size=(res, res), mode="area")[:, 0]
        out.append(cam.numpy().astype(np.float16))
    return np.concatenate(out)


def make_folds(y, pid, k, seed):
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    fold = np.zeros(len(y), dtype=int)
    for f, (_, te) in enumerate(sgkf.split(np.zeros(len(y)), y, groups=pid)):
        fold[te] = f
    return fold


def fold_split(y, pid, fold, f, seed):
    te = np.where(fold == f)[0]
    rest = np.where(fold != f)[0]
    pats = np.unique(pid[rest])
    plab = np.array([np.bincount(y[pid == p]).argmax() for p in pats])
    tr_p, va_p = train_test_split(pats, test_size=0.176, stratify=plab, random_state=seed)  # ~70/15/15 overall
    tr = rest[np.isin(pid[rest], tr_p)]; va = rest[np.isin(pid[rest], va_p)]
    assert not (set(pid[tr]) & set(pid[te])) and not (set(pid[va]) & set(pid[te])) and not (set(pid[tr]) & set(pid[va]))
    return tr, va, te


def leaky_split(y, n_tr, n_va, seed):
    ids = np.arange(len(y))
    tr, rest = train_test_split(ids, train_size=n_tr, stratify=y, random_state=seed)
    va, te = train_test_split(rest, train_size=n_va, stratify=y[rest], random_state=seed)
    return tr, va, te


def train_one(arch, tr, va, img, y, args, dev, log, seed):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model, _ = RE.build_model(arch, not args.no_pretrained)
    model = model.to(dev)
    cw = torch.tensor(len(tr) / (len(CLASSES) * np.bincount(y[tr], minlength=len(CLASSES))), dtype=torch.float32)
    dl_tr = DataLoader(RE.SliceDS(img, y, tr, True), batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       pin_memory=dev.type == "cuda")
    dl_va = DataLoader(RE.SliceDS(img, y, va, False), batch_size=args.batch, num_workers=args.workers)
    model, n_ep, best = RE.train_model(model, dl_tr, dl_va, cw, args, dev, log)
    return model, n_ep, best


# ------------------------------------------------------------------ analysis
def pearson_rows(a, b):
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    den = np.sqrt((a ** 2).sum(1) * (b ** 2).sum(1)) + 1e-8
    return (a * b).sum(1) / den


def top_iou(a, b, q=0.8):
    ta = a >= np.quantile(a, q, axis=1, keepdims=True); tb = b >= np.quantile(b, q, axis=1, keepdims=True)
    return (ta & tb).sum(1) / np.maximum((ta | tb).sum(1), 1)


def peak_xy(c):
    flat = c.reshape(c.shape[0], -1).argmax(1)
    return np.stack(np.unravel_index(flat, c.shape[1:]), 1)


def loc_metrics(cams, masks):
    """cams, masks: (N, h, w). returns hit (bool), energy ratio, IoU at 0.5."""
    pk = peak_xy(cams); n = len(cams)
    hit = masks[np.arange(n), pk[:, 0], pk[:, 1]].astype(bool)
    energy = (cams * masks).reshape(n, -1).sum(1) / (cams.reshape(n, -1).sum(1) + 1e-8)
    th = cams >= 0.5 * cams.reshape(n, -1).max(1)[:, None, None].clip(1e-8)
    iou = (th & masks.astype(bool)).reshape(n, -1).sum(1) / np.maximum((th | masks.astype(bool)).reshape(n, -1).sum(1), 1)
    return hit, energy, iou


def retrieval_same_patient(feats, y, pid):
    """Top-1 nearest neighbour (cosine), restricted to slices of the same tumour class, excluding the slice itself.
    Returns hit rate (neighbour from the same patient) and the chance rate."""
    f = feats.astype(np.float32); f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    hits, chance = [], []
    for k in np.unique(y):
        idx = np.where(y == k)[0]
        if len(idx) < 3:
            continue
        S = f[idx] @ f[idx].T; np.fill_diagonal(S, -np.inf)
        nn = idx[S.argmax(1)]
        hits.extend(pid[nn] == pid[idx])
        for i in idx:
            chance.append((np.sum(pid[idx] == pid[i]) - 1) / (len(idx) - 1))
    return float(np.mean(hits)), float(np.mean(chance))


def analyse(args, y, pid, mask_lr, fold, img):
    R = {}
    L = ["EXPLANATION MULTIPLICITY & LEAKAGE FINGERPRINT — RESULTS SUMMARY",
         f"folds: {args.folds} (patient-wise, StratifiedGroupKFold, seed {args.split_seed})   seeds: {args.seeds}   "
         f"Grad-CAM resolution for agreement: {args.cam_res}x{args.cam_res}", ""]
    fig_done = False
    for arch in args.archs:
        A = dict(acc=[], f1=[], pred_disagree=[], hit=[], hit_flip=[], pear=[], tiou=[], peak_mm=[], ens_hit=[],
                 ens_energy=[], mean_energy=[], pear_agree=[], pear_disagree=[], pat_range=[], n_allcorrect=[])
        per_fold_rows = []
        for f in range(args.folds):
            runs = [s for s in args.seeds if os.path.exists(run_path(args, arch, f, s))]
            if not runs:
                continue
            D = [np.load(run_path(args, arch, f, s)) for s in runs]
            te = D[0]["te"]; yt = y[te]; masks = mask_lr[te]
            P = np.stack([d["probs"] for d in D]); C = np.stack([d["cams"].astype(np.float32) for d in D])
            pred = P.argmax(2)
            accs = (pred == yt).mean(1)
            f1s = [json.load(open(run_path(args, arch, f, s).replace(".npz", ".json")))["macro_f1"] for s in runs]
            A["acc"] += list(accs); A["f1"] += f1s
            row = f"  fold {f}: n_test={len(te)}  acc per seed = " + ", ".join(f"{100 * a:.1f}" for a in accs)
            if len(runs) >= 2:
                disagree = (pred != pred[0]).any(0)
                allc = (pred == yt).all(0)
                A["pred_disagree"].append(disagree.mean()); A["n_allcorrect"].append(allc.sum())
                if allc.sum() < 2:
                    per_fold_rows.append(row + " | too few slices correct in every seed for explanation analysis")
                    continue
                hits = np.stack([loc_metrics(C[s], masks)[0] for s in range(len(runs))])          # (S, N)
                energies = np.stack([loc_metrics(C[s], masks)[1] for s in range(len(runs))])
                A["hit"].append(hits[:, allc].mean())
                flip = hits[:, allc].any(0) & ~hits[:, allc].all(0)
                A["hit_flip"].append(flip.mean())
                flat = C.reshape(len(runs), len(te), -1)
                pear = np.mean([pearson_rows(flat[a], flat[b]) for a, b in combinations(range(len(runs)), 2)], 0)
                tiou = np.mean([top_iou(flat[a], flat[b]) for a, b in combinations(range(len(runs)), 2)], 0)
                pk = np.stack([peak_xy(C[s]) for s in range(len(runs))])                           # (S, N, 2)
                scale = 512 / args.cam_res * PIX_MM
                pdist = np.mean([np.linalg.norm(pk[a] - pk[b], axis=1) for a, b in combinations(range(len(runs)), 2)], 0) * scale
                A["pear"].append(np.median(pear[allc])); A["tiou"].append(np.median(tiou[allc])); A["peak_mm"].append(np.median(pdist[allc]))
                A["pear_agree"] += list(pear[~disagree]); A["pear_disagree"] += list(pear[disagree])
                ens = C.mean(0)
                eh, ee, _ = loc_metrics(ens, masks)
                A["ens_hit"].append(eh[allc].mean()); A["ens_energy"].append(ee[allc].mean()); A["mean_energy"].append(energies[:, allc].mean())
                # per-patient hit rate range across seeds
                for p in np.unique(pid[te][allc]):
                    sel = allc & (pid[te] == p)
                    rates = hits[:, sel].mean(1)
                    A["pat_range"].append(rates.max() - rates.min())
                row += (f" | pred. disagreement {100 * disagree.mean():.1f}% | hit {100 * hits[:, allc].mean():.1f}% "
                        f"| hit flips {100 * flip.mean():.1f}% | median CAM r {np.median(pear[allc]):.2f} | ensemble hit {100 * eh[allc].mean():.1f}%")
                if not fig_done and arch == args.archs[0] and len(runs) >= 3:
                    j = np.where(allc)[0][np.argmin(pear[allc])]
                    fig_seeds(img[te[j]], C[:, j], ens[j], masks[j], runs, hits[:, j], os.path.join(args.out, "FigureM1_seed_heatmaps.png"))
                    fig_done = True
            per_fold_rows.append(row)
        if not A["acc"]:
            continue
        m = lambda v: f"{100 * np.nanmean(v):.1f} ± {100 * np.nanstd(v):.1f}" if len(v) else "n/a"
        L.append(f"=== {arch} ===")
        L += per_fold_rows
        L.append(f"Accuracy over all fold-seed runs: {m(A['acc'])} %   macro F1: {m(A['f1'])} %   (n runs = {len(A['acc'])})")
        if A["hit"] and A["pear_agree"]:
            p_mw = mannwhitneyu(A["pear_agree"], A["pear_disagree"]).pvalue if len(A["pear_disagree"]) > 1 else float("nan")
            L += [
                "TABLE M1 — explanation multiplicity (slices correctly classified by every seed; mean ± SD over folds)",
                f"  slices with any prediction disagreement between seeds: {m(A['pred_disagree'])} %",
                f"  pointing-game hit rate (single run): {m(A['hit'])} %",
                f"  slices whose hit/miss depends on the seed ('hit flips'): {m(A['hit_flip'])} %",
                f"  median pairwise Grad-CAM correlation between seeds: {np.mean(A['pear']):.3f} ± {np.std(A['pear']):.3f}",
                f"  median pairwise top-20% region IoU between seeds: {np.mean(A['tiou']):.3f} ± {np.std(A['tiou']):.3f}",
                f"  median distance between seeds' heatmap peaks: {np.mean(A['peak_mm']):.1f} ± {np.std(A['peak_mm']):.1f} mm",
                f"  ensemble (mean of seeds) heatmap hit rate: {m(A['ens_hit'])} %   vs single run {m(A['hit'])} %",
                f"  energy in mask: ensemble {np.mean(A['ens_energy']):.3f} vs single run {np.mean(A['mean_energy']):.3f}",
                f"  per-patient hit-rate range across seeds: median {np.median(A['pat_range']):.2f}, "
                f"patients with range >= 0.5: {100 * np.mean(np.array(A['pat_range']) >= 0.5):.1f} %",
                f"  Grad-CAM correlation, slices where seeds agree on class vs disagree: median {np.median(A['pear_agree']):.3f} "
                f"vs {np.median(A['pear_disagree']) if A['pear_disagree'] else float('nan'):.3f} (Mann-Whitney P = {p_mw:.2g})",
                ""]
            if arch == args.archs[0]:
                fig_hist(A["pear_agree"], A["pear_disagree"], os.path.join(args.out, "FigureM2_agreement_hist.png"))
        R[arch] = {k: (np.asarray(v, dtype=float).tolist() if k not in ("pear_agree", "pear_disagree", "pat_range") else None)
                   for k, v in A.items()}

    # ---- leakage fingerprint
    LK = []
    for f in range(args.folds):
        p = os.path.join(args.out, "runs", f"leaky_f{f}.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p); te = d["probe"]
        r = dict(fold=f, leaky_own_acc=float(d["leaky_own_acc"]), leaky_probe_acc=float(d["leaky_probe_acc"]))
        for name in ("pixel", "imagenet", "clean", "leaky"):
            if name in d.files:
                r[name], r["chance"] = retrieval_same_patient(d[name], y[te], pid[te])
        if os.path.exists(run_path(args, "resnet50", f, args.seeds[0])):
            r["clean_acc"] = float(json.load(open(run_path(args, "resnet50", f, args.seeds[0]).replace(".npz", ".json")))["accuracy"])
        LK.append(r)
    if LK:
        L.append("TABLE M2 — leakage fingerprint (ResNet-50, seed 0). Probe set = clean test patients of each fold.")
        L.append("  Same-patient retrieval: nearest neighbour of a slice (same tumour class, cosine) comes from the same patient.")
        L.append("  fold | clean acc (patient split) | leaky acc (own slice-level test) | leaky acc on probe set | "
                 "re-ID: chance | pixels | ImageNet | clean model | leaky model")
        for r in LK:
            L.append(f"  {r['fold']} | {100 * r.get('clean_acc', float('nan')):.1f} | {100 * r['leaky_own_acc']:.1f} | "
                     f"{100 * r['leaky_probe_acc']:.1f} | {100 * r['chance']:.1f} | {100 * r['pixel']:.1f} | "
                     f"{100 * r['imagenet']:.1f} | {100 * r.get('clean', float('nan')):.1f} | {100 * r['leaky']:.1f}")
        mm = lambda k: f"{100 * np.mean([r[k] for r in LK if k in r]):.1f} ± {100 * np.std([r[k] for r in LK if k in r]):.1f}"
        L.append(f"  mean | {mm('clean_acc')} | {mm('leaky_own_acc')} | {mm('leaky_probe_acc')} | {mm('chance')} | "
                 f"{mm('pixel')} | {mm('imagenet')} | {mm('clean')} | {mm('leaky')}")
        fig_reid(LK, os.path.join(args.out, "FigureM3_reidentification.png"))
        R["leakage"] = LK
    json.dump(R, open(os.path.join(args.out, "multiplicity_results.json"), "w"), indent=1, default=float)
    open(os.path.join(args.out, "multiplicity_summary.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


# ------------------------------------------------------------------ figures
def fig_seeds(im, cams, ens, mask, seeds, hits, path):
    n = len(cams) + 2
    fig, ax = plt.subplots(1, n, figsize=(2.3 * n, 2.6), dpi=200)
    im = im.astype(np.float32)
    up = lambda c: F.interpolate(torch.from_numpy(np.asarray(c, dtype=np.float32))[None, None], size=im.shape, mode="bilinear")[0, 0].numpy()
    mk = F.interpolate(torch.from_numpy(mask.astype(np.float32))[None, None], size=im.shape, mode="nearest")[0, 0].numpy()
    ax[0].imshow(im, cmap="gray"); ax[0].contour(mk, levels=[0.5], colors="yellow", linewidths=0.8); ax[0].set_title("Slice + mask", fontsize=8)
    for i, (c, s) in enumerate(zip(cams, seeds)):
        ax[i + 1].imshow(im, cmap="gray"); ax[i + 1].imshow(up(c), cmap="jet", alpha=0.45)
        ax[i + 1].contour(mk, levels=[0.5], colors="white", linewidths=0.8)
        ax[i + 1].set_title(f"Seed {s} ({'hit' if hits[i] else 'miss'})", fontsize=8)
    ax[-1].imshow(im, cmap="gray"); ax[-1].imshow(up(ens), cmap="jet", alpha=0.45)
    ax[-1].contour(mk, levels=[0.5], colors="white", linewidths=0.8); ax[-1].set_title("Ensemble", fontsize=8)
    for a in ax: a.axis("off")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_hist(agree, disagree, path):
    fig, ax = plt.subplots(figsize=(5, 3.4), dpi=200)
    bins = np.linspace(-0.2, 1, 37)
    ax.hist(agree, bins=bins, alpha=0.7, density=True, label="Seeds agree on class")
    if len(disagree):
        ax.hist(disagree, bins=bins, alpha=0.7, density=True, label="Seeds disagree on class")
    ax.set_xlabel("Mean pairwise Grad-CAM correlation between seeds"); ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_reid(LK, path):
    keys = [("chance", "Chance"), ("pixel", "Raw pixels"), ("imagenet", "ImageNet\n(no fine-tuning)"),
            ("clean", "Clean model\n(patient split)"), ("leaky", "Leaky model\n(slice split)")]
    vals = [[100 * r[k] for r in LK if k in r] for k, _ in keys]
    fig, ax = plt.subplots(figsize=(6, 3.6), dpi=200)
    ax.bar(range(len(keys)), [np.mean(v) for v in vals], yerr=[np.std(v) for v in vals], capsize=4,
           color=["#BBBBBB", "#9DB9D3", "#9DB9D3", "#2F5D8A", "#C0504D"])
    for i, v in enumerate(vals):
        ax.text(i, np.mean(v) + np.std(v) + 1.5, f"{np.mean(v):.1f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(keys)), [k[1] for k in keys], fontsize=8); ax.set_ylabel("Same-patient retrieval (%)")
    ax.set_ylim(0, 105); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# ------------------------------------------------------------------ main
def run_path(args, arch, f, s):
    return os.path.join(args.out, "runs", f"{arch}_f{f}_s{s}.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/figshare_brain_tumour")
    ap.add_argument("--out", default="results_mult")
    ap.add_argument("--cache", default=None, help="existing cache.npz to reuse (e.g. results_v2/cache.npz)")
    ap.add_argument("--archs", nargs="+", default=["resnet50"], choices=["resnet50", "densenet121"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--no_leaky", action="store_true", help="skip the leakage-fingerprint runs")
    ap.add_argument("--analyse_only", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--cam_res", type=int, default=128)
    ap.add_argument("--no_pretrained", action="store_true", help="for offline testing only")
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "runs"), exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logf = open(os.path.join(args.out, "log.txt"), "a")
    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    log(f"Device: {dev} | torch {torch.__version__}")

    cache = args.cache or os.path.join(args.out, "cache.npz")
    if not os.path.exists(cache) and RE.count_mat(args.data_dir) == 0:
        RE.download_dataset(args.data_dir)
    img, mask, y, pid = RE.load_dataset(args.data_dir, cache)
    f_ = args.cam_res
    mask_lr = F.max_pool2d(torch.from_numpy(mask.astype(np.float32))[:, None], kernel_size=512 // f_)[:, 0].numpy().astype(np.uint8)
    fold = make_folds(y, pid, args.folds, args.split_seed)
    log(f"Dataset: {len(y)} slices, {len(np.unique(pid))} patients; fold sizes (patients): "
        f"{[len(np.unique(pid[fold == k])) for k in range(args.folds)]}")

    if not args.analyse_only:
        for f in range(args.folds):
            tr, va, te = fold_split(y, pid, fold, f, args.split_seed)
            ds_te = RE.SliceDS(img, y, te, False)
            dl_te = DataLoader(ds_te, batch_size=args.batch, num_workers=args.workers)
            for arch in args.archs:
                for s in args.seeds:
                    p = run_path(args, arch, f, s)
                    if os.path.exists(p):
                        log(f"[{arch} fold {f} seed {s}] done already, skipping"); continue
                    log(f"\n=== {arch} | fold {f} | seed {s} | train {len(tr)} val {len(va)} test {len(te)} slices ===")
                    t0 = time.time()
                    model, n_ep, best = train_one(arch, tr, va, img, y, args, dev, log, s)
                    probs, feats = predict_with_features(model, arch, dl_te, dev)
                    cams = gradcam_maps(model, arch, ds_te, probs.argmax(1), dev, args.cam_res)
                    m = RE.metrics(y[te], probs)
                    log(f"    test acc {m['accuracy']:.4f}  macroF1 {m['macro_f1']:.4f}  epochs {n_ep}  {time.time() - t0:.0f}s")
                    save = dict(te=te, probs=probs.astype(np.float32), cams=cams)
                    if s == args.seeds[0]:
                        save["feats"] = feats
                    np.savez_compressed(p, **save)
                    json.dump(dict(accuracy=m["accuracy"], macro_f1=m["macro_f1"], epochs=n_ep, val_loss=best),
                              open(p.replace(".npz", ".json"), "w"))
                    del model; torch.cuda.empty_cache() if dev.type == "cuda" else None
            # ---- leakage fingerprint for this fold
            lp = os.path.join(args.out, "runs", f"leaky_f{f}.npz")
            if not args.no_leaky and not os.path.exists(lp) and os.path.exists(run_path(args, "resnet50", f, args.seeds[0])):
                log(f"\n=== leaky ResNet-50 | fold {f} (slice-level split, same set sizes) ===")
                ltr, lva, lte = leaky_split(y, len(tr), len(va), args.split_seed + f)
                model, n_ep, best = train_one("resnet50", ltr, lva, img, y, args, dev, log, args.seeds[0])
                p_own, _ = predict_with_features(model, "resnet50", DataLoader(RE.SliceDS(img, y, lte, False),
                                                 batch_size=args.batch, num_workers=args.workers), dev)
                p_probe, f_leaky = predict_with_features(model, "resnet50", dl_te, dev)
                del model
                base, _ = RE.build_model("resnet50", not args.no_pretrained)
                _, f_imnet = predict_with_features(base.to(dev), "resnet50", dl_te, dev)
                del base
                pix = F.interpolate(torch.from_numpy(img[te].astype(np.float32))[:, None], size=(32, 32), mode="area").flatten(1).numpy()
                clean = np.load(run_path(args, "resnet50", f, args.seeds[0]))["feats"]
                own_acc = float((p_own.argmax(1) == y[lte]).mean()); probe_acc = float((p_probe.argmax(1) == y[te]).mean())
                log(f"    leaky acc on own slice-level test {own_acc:.4f}  on clean probe set {probe_acc:.4f}  "
                    f"({100 * np.isin(pid[te], pid[ltr]).mean():.0f}% of probe slices come from patients in its training set)")
                np.savez_compressed(lp, probe=te, leaky=f_leaky, imagenet=f_imnet, clean=clean, pixel=pix.astype(np.float16),
                                    leaky_own_acc=own_acc, leaky_probe_acc=probe_acc)
                torch.cuda.empty_cache() if dev.type == "cuda" else None
    analyse(args, y, pid, mask_lr, fold, img)
    log(f"\nDone. Send {os.path.join(args.out, 'multiplicity_summary.txt')} and the FigureM*.png files.")


if __name__ == "__main__":
    main()
