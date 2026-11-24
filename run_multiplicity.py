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
import argparse, copy, glob, json, os, random, sys, time
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


METHODS = ["gradcam", "gradcampp", "hirescam"]


class MultiCAM:
    """Grad-CAM, Grad-CAM++ and HiResCAM from one forward/backward pass on the same target layer."""
    def __init__(self, model, layer):
        self.model, self.A, self.G = model, None, None
        def fwd(m, i, o):
            self.A = o
            if o.requires_grad:
                o.register_hook(lambda g: setattr(self, "G", g))
        self.h = layer.register_forward_hook(fwd)

    def __call__(self, x, cls, res):
        self.model.zero_grad(set_to_none=True)
        out = self.model(x)
        out[torch.arange(len(cls)), cls].sum().backward()
        A, G = self.A.detach().float(), self.G.detach().float()
        maps = {}
        maps["gradcam"] = F.relu((G.mean((2, 3), keepdim=True) * A).sum(1, keepdim=True))
        g2, g3 = G ** 2, G ** 3
        alpha = g2 / (2 * g2 + A.sum((2, 3), keepdim=True) * g3 + 1e-7)
        alpha = torch.where(G != 0, alpha, torch.zeros_like(alpha))
        w = (F.relu(G) * alpha).sum((2, 3), keepdim=True)
        maps["gradcampp"] = F.relu((w * A).sum(1, keepdim=True))
        maps["hirescam"] = F.relu((G * A).sum(1, keepdim=True))
        res_maps = {}
        for k, c in maps.items():
            c = F.interpolate(c, size=(res, res), mode="bilinear", align_corners=False)[:, 0]
            c = c / (c.flatten(1).amax(1).view(-1, 1, 1) + 1e-8)
            res_maps[k] = c.cpu().numpy().astype(np.float16)
        return res_maps

    def remove(self):
        self.h.remove()


def multi_cam_maps(model, arch, ds, pred, dev, res, bs=16, methods=METHODS):
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    mc = MultiCAM(model, target_layer(model, arch))
    out = {m: [] for m in methods}
    for s in range(0, len(ds), bs):
        ii = list(range(s, min(s + bs, len(ds))))
        x = torch.stack([ds[i][0] for i in ii]).to(dev)
        r = mc(x, torch.tensor(pred[ii], device=dev), res)
        for m in methods:
            out[m].append(r[m])
    mc.remove()
    return {m: np.concatenate(v) for m, v in out.items()}


def randomised_copy(model, arch):
    """Model-parameter randomisation test (Adebayo et al.): re-initialise the explained block."""
    m2 = copy.deepcopy(model)
    blk = m2.layer4 if arch == "resnet50" else m2.features.denseblock4
    for mod in blk.modules():
        if hasattr(mod, "reset_parameters") and mod is not blk:
            mod.reset_parameters()
    if arch == "densenet121":
        m2.features.norm5.reset_parameters()
    return m2


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


def load_cams(d, method):
    k = f"cam_{method}"
    if k in d.files:
        return d[k].astype(np.float32)
    if method == "gradcam" and "cams" in d.files:  # results from version 1
        return d["cams"].astype(np.float32)
    return None


def multiplicity_block(C, masks, allc, res):
    """C: (S, N, h, w) maps of S seeds; masks (N, h, w); allc: bool (N,). Returns dict of metrics on allc slices."""
    S = len(C)
    hits = np.stack([loc_metrics(C[s], masks)[0] for s in range(S)])
    energies = np.stack([loc_metrics(C[s], masks)[1] for s in range(S)])
    flat = C.reshape(S, C.shape[1], -1)
    pairs = list(combinations(range(S), 2))
    pear = np.mean([pearson_rows(flat[a], flat[b]) for a, b in pairs], 0)
    tiou = np.mean([top_iou(flat[a], flat[b]) for a, b in pairs], 0)
    pk = np.stack([peak_xy(C[s]) for s in range(S)])
    pdist = np.mean([np.linalg.norm(pk[a] - pk[b], axis=1) for a, b in pairs], 0) * (512 / res * PIX_MM)
    ens = C.mean(0); eh, ee, _ = loc_metrics(ens, masks)
    h = hits[:, allc]
    # consensus regions: pixels in the top-20% region of >= k seeds
    top = flat >= np.quantile(flat, 0.8, axis=2, keepdims=True)                 # (S, N, P)
    votes = top.sum(0)                                                          # (N, P)
    mflat = masks.reshape(len(masks), -1).astype(bool)
    def prec(region):
        n = region.sum(1)
        return np.where(n > 0, (region & mflat).sum(1) / np.maximum(n, 1), np.nan)
    prec_single = np.nanmean(np.stack([prec(top[s_]) for s_ in range(S)]), 0)
    prec_major = prec(votes >= (S // 2 + 1)); prec_all = prec(votes >= S)
    c = C.shape[-1] // 2
    rec = dict(hit=h.mean(0), flip=(h.any(0) & ~h.all(0)).astype(float), all_hit=h.all(0).astype(float),
               ens_hit=eh[allc].astype(float), centre=masks[allc, c, c].astype(float),
               area=mflat[allc].mean(1), prec_single=prec_single[allc], prec_major=prec_major[allc],
               prec_all=prec_all[allc], empty_all=(votes[allc] >= S).sum(1) == 0)
    return dict(_slice=rec,hit=h.mean(), hit_flip=(h.any(0) & ~h.all(0)).mean(), all_hit=h.all(0).mean(),
                pear=np.median(pear[allc]), tiou=np.median(tiou[allc]), peak_mm=np.median(pdist[allc]),
                ens_hit=eh[allc].mean(), ens_energy=ee[allc].mean(), energy=energies[:, allc].mean(),
                _pear_all=pear, _hits=hits, _ens=ens)


def patient_bootstrap(rec, keys, n_boot=2000, seed=0):
    """rec: dict of per-slice arrays incl. 'pid'. Returns {key: (mean, lo, hi)} with patient-level resampling."""
    rng = np.random.default_rng(seed)
    pats = np.unique(rec["pid"]); groups = [np.where(rec["pid"] == p)[0] for p in pats]
    est = {k: float(np.nanmean(rec[k])) for k in keys}
    B = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = np.concatenate([groups[i] for i in rng.integers(0, len(pats), len(pats))])
        for k in keys:
            B[k].append(np.nanmean(rec[k][idx]))
    return {k: (est[k], float(np.nanpercentile(B[k], 2.5)), float(np.nanpercentile(B[k], 97.5))) for k in keys}


def boot_p(rec, key, n_boot=2000, seed=1):
    """Two-sided bootstrap P for mean(rec[key]) == 0 (patient resampling)."""
    rng = np.random.default_rng(seed)
    pats = np.unique(rec["pid"]); groups = [np.where(rec["pid"] == p)[0] for p in pats]
    b = np.array([np.nanmean(rec[key][np.concatenate([groups[i] for i in rng.integers(0, len(pats), len(pats))])])
                  for _ in range(n_boot)])
    return float(min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))), n_boot


def ci_table(per_method, args):
    L = ["TABLE M3 — pooled over folds, 95% CIs from 2,000 patient-level bootstrap resamples (slices correct in every seed)",
         "  method | n slices / patients | hit rate % | hit flips % | hit by all seeds % | ensemble hit % | image-centre hit % | "
         "hit minus centre, pp [P] | ensemble minus single, pp [P] | top-20% precision: single / majority / unanimous | random-region precision | unanimous region empty %"]
    f = lambda t, mul=100: f"{mul * t[0]:.1f} ({mul * t[1]:.1f}–{mul * t[2]:.1f})"
    for m, rs in per_method.items():
        if not rs:
            continue
        rec = {k: np.concatenate([r["_slice"][k] for r in rs]) for k in rs[0]["_slice"]}
        if len(rec["pid"]) < 2:
            L.append(f"  {m} | too few slices"); continue
        rec["d_centre"] = rec["hit"] - rec["centre"]; rec["d_ens"] = rec["ens_hit"] - rec["hit"]
        rec["d_cons"] = rec["prec_major"] - rec["prec_single"]
        ci = patient_bootstrap(rec, ["hit", "flip", "all_hit", "ens_hit", "centre", "d_centre", "d_ens",
                                     "prec_single", "prec_major", "prec_all", "area", "d_cons"])
        p_c, nb = boot_p(rec, "d_centre"); p_e, _ = boot_p(rec, "d_ens"); p_k, _ = boot_p(rec, "d_cons")
        fp = lambda p: f"<{1 / nb:.4f}" if p == 0 else f"{p:.4f}"
        L.append(f"  {m} | {len(rec['pid'])} / {len(np.unique(rec['pid']))} | {f(ci['hit'])} | {f(ci['flip'])} | {f(ci['all_hit'])} | "
                 f"{f(ci['ens_hit'])} | {f(ci['centre'])} | {f(ci['d_centre'])} [{fp(p_c)}] | {f(ci['d_ens'])} [{fp(p_e)}] | "
                 f"{100 * ci['prec_single'][0]:.1f} / {100 * ci['prec_major'][0]:.1f} / {100 * ci['prec_all'][0]:.1f} "
                 f"(majority minus single {f(ci['d_cons'])} pp [{fp(p_k)}]) | {100 * ci['area'][0]:.1f} | {100 * np.mean(rec['empty_all']):.1f}")
    return L


def analyse(args, y, pid, mask_lr, fold, img):
    R = {"methods": {}, "baselines": [], "randomisation": {}, "cross_arch": []}
    res = args.cam_res
    L = ["EXPLANATION MULTIPLICITY — RESULTS SUMMARY (version 2)",
         f"folds: {args.folds} (patient-wise, StratifiedGroupKFold, seed {args.split_seed})   seeds: {args.seeds}   "
         f"map resolution for analysis: {res}x{res}", ""]
    fig_done = False
    allc_by = {}          # (arch, fold) -> allc mask, te
    maps_by = {}          # (arch, fold) -> gradcam maps (S, N, h, w)
    for arch in args.archs:
        accs, f1s, disagrees, rows = [], [], [], []
        per_method = {m: [] for m in METHODS}
        pear_agree, pear_dis = [], []
        rand_corr, rand_hit, real_hit0 = [], [], []
        for f in range(args.folds):
            runs = [s for s in args.seeds if os.path.exists(run_path(args, arch, f, s))]
            if not runs:
                continue
            D = [np.load(run_path(args, arch, f, s)) for s in runs]
            te = D[0]["te"]; yt = y[te]; masks = mask_lr[te]
            P = np.stack([d["probs"] for d in D]); pred = P.argmax(2)
            accs += list((pred == yt).mean(1))
            f1s += [json.load(open(run_path(args, arch, f, s).replace(".npz", ".json")))["macro_f1"] for s in runs]
            row = f"  fold {f}: n_test={len(te)}  acc per seed = " + ", ".join(f"{100 * a:.1f}" for a in (pred == yt).mean(1))
            # baselines (independent of model): random point = mask area fraction; centre pixel
            if arch == args.archs[0]:
                c = res // 2
                R["baselines"].append(dict(fold=f, random_point=float(masks.reshape(len(te), -1).mean()),
                                           centre=float(masks[:, c, c].mean())))
            # randomisation test (seed 0)
            if "cam_rand" in D[0].files:
                g0 = load_cams(D[0], "gradcam"); gr = D[0]["cam_rand"].astype(np.float32)
                ok0 = pred[0] == yt
                rand_corr.append(np.median(pearson_rows(g0.reshape(len(te), -1), gr.reshape(len(te), -1))[ok0]))
                rand_hit.append(loc_metrics(gr, masks)[0][ok0].mean()); real_hit0.append(loc_metrics(g0, masks)[0][ok0].mean())
            if len(runs) < 2:
                rows.append(row); continue
            disagree = (pred != pred[0]).any(0); allc = (pred == yt).all(0)
            disagrees.append(disagree.mean())
            if allc.sum() < 2:
                rows.append(row + " | too few slices correct in every seed"); continue
            allc_by[(arch, f)] = (allc, te)
            for m in METHODS:
                C = [load_cams(d, m) for d in D]
                if any(c is None for c in C):
                    continue
                C = np.stack(C)
                r = multiplicity_block(C, masks, allc, res)
                r["_slice"]["pid"] = pid[te][allc]
                per_method[m].append(r)
                if m == "gradcam":
                    maps_by[(arch, f)] = (C, runs)
                    pear_agree += list(r["_pear_all"][~disagree]); pear_dis += list(r["_pear_all"][disagree])
                    row += (f" | disagree {100 * disagree.mean():.1f}% | Grad-CAM hit {100 * r['hit']:.1f}% flips {100 * r['hit_flip']:.1f}% "
                            f"r {r['pear']:.2f} | ensemble {100 * r['ens_hit']:.1f}%")
                    if not fig_done and len(runs) >= 3:
                        j = np.where(allc)[0][np.argmin(r["_pear_all"][allc])]
                        fig_seeds(img[te[j]], C[:, j], r["_ens"][j], masks[j], runs, r["_hits"][:, j],
                                  os.path.join(args.out, "FigureM1_seed_heatmaps.png"))
                        fig_done = True
            rows.append(row)
        if not accs:
            continue
        m_ = lambda v: f"{100 * np.nanmean(v):.1f} ± {100 * np.nanstd(v):.1f}" if len(v) else "n/a"
        L.append(f"=== {arch} ===")
        L += rows
        L.append(f"Accuracy over all fold-seed runs: {m_(accs)} %   macro F1: {m_(f1s)} %   (n runs = {len(accs)})")
        L.append(f"Slices with prediction disagreement between seeds: {m_(disagrees)} %")
        L.append("TABLE M1 — explanation multiplicity by method (slices correct in every seed; mean ± SD over folds)")
        L.append("  method | hit rate % | hit flips % | hit by all seeds % | median map r | top-20% IoU | peak distance mm | ensemble hit % | energy in mask (single / ensemble)")
        R["methods"][arch] = {}
        for m in METHODS:
            rs = per_method[m]
            if not rs:
                continue
            g = lambda k: [r[k] for r in rs]
            L.append(f"  {m} | {m_(g('hit'))} | {m_(g('hit_flip'))} | {m_(g('all_hit'))} | {np.mean(g('pear')):.3f} ± {np.std(g('pear')):.3f} | "
                     f"{np.mean(g('tiou')):.3f} | {np.mean(g('peak_mm')):.1f} ± {np.std(g('peak_mm')):.1f} | {m_(g('ens_hit'))} | "
                     f"{np.mean(g('energy')):.3f} / {np.mean(g('ens_energy')):.3f}")
            R["methods"][arch][m] = {k: [float(r[k]) for r in rs] for k in rs[0] if not k.startswith("_")}
        L += ci_table(per_method, args)
        if pear_dis:
            p_mw = mannwhitneyu(pear_agree, pear_dis).pvalue
            L.append(f"  Grad-CAM map r, seeds agree vs disagree on class: {np.median(pear_agree):.3f} vs {np.median(pear_dis):.3f} (Mann-Whitney P = {p_mw:.2g})")
            if arch == args.archs[0]:
                fig_hist(pear_agree, pear_dis, os.path.join(args.out, "FigureM2_agreement_hist.png"))
        if rand_corr:
            L.append(f"  Randomisation test (seed 0, last block re-initialised): median r(trained, randomised) = "
                     f"{np.mean(rand_corr):.3f} ± {np.std(rand_corr):.3f}; hit rate randomised {m_(rand_hit)} % vs trained {m_(real_hit0)} %")
            R["randomisation"][arch] = dict(corr=rand_corr, rand_hit=rand_hit, real_hit=real_hit0)
        L.append("")
    if R["baselines"]:
        b = R["baselines"]
        L.append(f"Chance baselines (all test slices): random point inside mask {100 * np.mean([x['random_point'] for x in b]):.1f} %, "
                 f"image-centre pixel inside mask {100 * np.mean([x['centre'] for x in b]):.1f} %")
    # cross-architecture agreement (Grad-CAM), slices correct for every seed of both architectures
    if len(args.archs) == 2:
        cr, cd, wr = [], [], []
        for f in range(args.folds):
            k0, k1 = (args.archs[0], f), (args.archs[1], f)
            if k0 not in maps_by or k1 not in maps_by:
                continue
            both = allc_by[k0][0] & allc_by[k1][0]
            if both.sum() < 2:
                continue
            C0, C1 = maps_by[k0][0][:, both], maps_by[k1][0][:, both]
            n = min(len(C0), len(C1))
            cr.append(np.median(np.mean([pearson_rows(C0[i].reshape(both.sum(), -1), C1[i].reshape(both.sum(), -1)) for i in range(n)], 0)))
            cd.append(np.median(np.mean([np.linalg.norm(peak_xy(C0[i]) - peak_xy(C1[i]), axis=1) for i in range(n)], 0) * (512 / res * PIX_MM)))
        if cr:
            L.append(f"Cross-architecture Grad-CAM agreement ({args.archs[0]} vs {args.archs[1]}, same seed index): "
                     f"median r {np.mean(cr):.3f} ± {np.std(cr):.3f}, peak distance {np.mean(cd):.1f} ± {np.std(cd):.1f} mm")
            R["cross_arch"] = dict(corr=cr, peak_mm=cd)
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
    ap.add_argument("--out", default="results_mult2")
    ap.add_argument("--cache", default=None, help="existing cache.npz to reuse (e.g. results_v2/cache.npz)")
    ap.add_argument("--archs", nargs="+", default=["resnet50", "densenet121"], choices=["resnet50", "densenet121"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--leaky", action="store_true", help="also run the leakage-fingerprint models")
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
                    cams = multi_cam_maps(model, arch, ds_te, probs.argmax(1), dev, args.cam_res)
                    m = RE.metrics(y[te], probs)
                    log(f"    test acc {m['accuracy']:.4f}  macroF1 {m['macro_f1']:.4f}  epochs {n_ep}  {time.time() - t0:.0f}s")
                    save = dict(te=te, probs=probs.astype(np.float32), **{f"cam_{k}": v for k, v in cams.items()})
                    if s == args.seeds[0]:
                        mr = randomised_copy(model, arch)
                        save["cam_rand"] = multi_cam_maps(mr, arch, ds_te, probs.argmax(1), dev, args.cam_res, methods=["gradcam"])["gradcam"]
                        del mr
                    if s == args.seeds[0]:
                        save["feats"] = feats
                    np.savez_compressed(p, **save)
                    json.dump(dict(accuracy=m["accuracy"], macro_f1=m["macro_f1"], epochs=n_ep, val_loss=best),
                              open(p.replace(".npz", ".json"), "w"))
                    del model; torch.cuda.empty_cache() if dev.type == "cuda" else None
            # ---- leakage fingerprint for this fold
            lp = os.path.join(args.out, "runs", f"leaky_f{f}.npz")
            if args.leaky and not os.path.exists(lp) and os.path.exists(run_path(args, "resnet50", f, args.seeds[0])):
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
