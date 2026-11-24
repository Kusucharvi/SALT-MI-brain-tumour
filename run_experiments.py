#!/usr/bin/env python3
"""
Brain tumour classification experiments for the paper
"Explainable Attention-Augmented ResNet-50 for Brain Tumour Classification on MRI
 with Quantitative Grad-CAM Evaluation".

Dataset: Cheng et al. figshare brain tumour dataset (3,064 T1-CE slices, 233 patients,
glioma / meningioma / pituitary), .mat (v7.3) files with image, label, PID, tumorMask.

Runs (each with the same patient-level split):
  proposed        ResNet-50 + channel & spatial attention (CBAM-style) after every stage
  resnet50        ResNet-50 without attention
  densenet121, efficientnet_b0, vit_b_16   baselines
  cbam_channel    ResNet-50 + channel attention only   (ablation)
  cbam_spatial    ResNet-50 + spatial attention only   (ablation)
  proposed_slicesplit   proposed model with a random slice-level split (leakage demo)

Outputs (in --out): results_summary.txt (send this), results.json, CSV tables, figures.

Usage (the dataset is downloaded automatically on the first run, ~880 MB):
  python run_experiments.py
  python run_experiments.py --seeds 0 1 2 3 4          # 5 seeds, as described in the paper
  python run_experiments.py --data_dir D:/my_copy     # use an already-downloaded copy
"""
import argparse, glob, json, os, random, shutil, time, urllib.request, zipfile
from collections import defaultdict

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.transforms import v2 as T
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from scipy.stats import binomtest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["Glioma", "Meningioma", "Pituitary"]
# figshare labels: 1 = meningioma, 2 = glioma, 3 = pituitary
LABEL_MAP = {1: 1, 2: 0, 3: 2}
RUNS_DEFAULT = ["proposed", "resnet50", "densenet121", "efficientnet_b0", "vit_b_16",
                "cbam_channel", "cbam_spatial", "proposed_slicesplit"]
# v2: identity-initialised (gated) attention variants
RUNS_GATED = ["gated", "gated_channel", "gated_spatial", "gated_slicesplit"]
RUNS_ALL = RUNS_DEFAULT + RUNS_GATED
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ------------------------------------------------------------------ download
FIGSHARE_ARTICLE = 1512427  # Cheng J. Brain tumor dataset. figshare. doi:10.6084/m9.figshare.1512427
FIGSHARE_PAGE = "https://figshare.com/articles/dataset/brain_tumor_dataset/1512427"


def _fetch(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research script)"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest + ".part", "wb") as f:
        total = int(r.headers.get("Content-Length", 0)); done = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk); done += len(chunk)
            if total:
                print(f"\r    {os.path.basename(dest)}: {done / 2**20:.0f} / {total / 2**20:.0f} MB", end="", flush=True)
    print()
    os.replace(dest + ".part", dest)


def _safe_extract(zpath, out):
    root = os.path.realpath(out)
    with zipfile.ZipFile(zpath) as z:
        for m in z.infolist():
            target = os.path.realpath(os.path.join(out, m.filename))
            if not target.startswith(root + os.sep) and target != root:
                raise SystemExit(f"Unsafe path in zip: {m.filename}")
        z.extractall(out)


def count_mat(d):
    return len([f for f in glob.glob(os.path.join(d, "**", "*.mat"), recursive=True)
                if os.path.basename(f).lower() != "cvind.mat"])


def download_dataset(data_dir):
    """Download the figshare brain tumour dataset (~880 MB) and unzip it into data_dir."""
    os.makedirs(data_dir, exist_ok=True)
    print(f"Downloading dataset from figshare (article {FIGSHARE_ARTICLE}) into {data_dir} ...")
    try:
        req = urllib.request.Request(f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE}/files",
                                     headers={"User-Agent": "Mozilla/5.0 (research script)"})
        files = json.load(urllib.request.urlopen(req, timeout=60))
    except Exception as e:
        raise SystemExit(f"Could not reach figshare ({e}).\nDownload the zip files manually from {FIGSHARE_PAGE}, "
                         f"unzip them into {data_dir}, and run again.")
    zdir = os.path.join(data_dir, "_zips"); os.makedirs(zdir, exist_ok=True)
    for fi in files:
        dest = os.path.join(zdir, fi["name"])
        if not (os.path.exists(dest) and os.path.getsize(dest) == fi.get("size", -1)):
            _fetch(fi["download_url"], dest)
        if dest.lower().endswith(".zip"):
            print(f"    unzipping {fi['name']}"); _safe_extract(dest, data_dir)
        else:
            shutil.copy(dest, os.path.join(data_dir, fi["name"]))
    # some releases contain zips inside zips
    for z in glob.glob(os.path.join(data_dir, "**", "*.zip"), recursive=True):
        if os.path.dirname(os.path.realpath(z)) != os.path.realpath(zdir):
            print(f"    unzipping {os.path.basename(z)}"); _safe_extract(z, os.path.dirname(z)); os.remove(z)
    n = count_mat(data_dir)
    print(f"Dataset ready: {n} .mat slice files.")
    if n < 3000:
        print("WARNING: expected 3,064 files. Check the download, or download manually from " + FIGSHARE_PAGE)
    shutil.rmtree(zdir, ignore_errors=True)


# ------------------------------------------------------------------ data
def load_dataset(data_dir, cache):
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        print(f"Loaded cache {cache}")
        return d["img"], d["mask"], d["y"], d["pid"]
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.mat"), recursive=True))
    files = [f for f in files if os.path.basename(f).lower() != "cvind.mat"]
    if not files:
        raise SystemExit(f"No .mat files found under {data_dir}")
    imgs, masks, ys, pids = [], [], [], []
    for i, f in enumerate(files):
        with h5py.File(f, "r") as h:
            c = h["cjdata"]
            im = np.array(c["image"], dtype=np.float32).T
            mk = np.array(c["tumorMask"], dtype=np.uint8).T
            lab = int(np.array(c["label"]).flatten()[0])
            pid = "".join(chr(int(v)) for v in np.array(c["PID"]).flatten() if int(v) != 0).strip()
        if im.shape != (512, 512):  # a few slices are 256x256
            im = F.interpolate(torch.from_numpy(im)[None, None], size=(512, 512), mode="bilinear",
                               align_corners=False)[0, 0].numpy()
            mk = (F.interpolate(torch.from_numpy(mk.astype(np.float32))[None, None], size=(512, 512),
                                mode="nearest")[0, 0].numpy() > 0.5).astype(np.uint8)
        im = (im - im.min()) / (im.max() - im.min() + 1e-8)  # min-max to [0,1]
        imgs.append(im.astype(np.float16)); masks.append(mk); ys.append(LABEL_MAP[lab]); pids.append(pid)
        if (i + 1) % 500 == 0:
            print(f"  read {i + 1}/{len(files)} files")
    img, mask, y, pid = np.stack(imgs), np.stack(masks), np.array(ys), np.array(pids)
    np.savez(cache, img=img, mask=mask, y=y, pid=pid)
    return img, mask, y, pid


def patient_split(y, pid, seed):
    pats = np.unique(pid)
    pat_lab = np.array([np.bincount(y[pid == p]).argmax() for p in pats])
    tr, rest, _, rest_lab = train_test_split(pats, pat_lab, test_size=0.30, stratify=pat_lab, random_state=seed)
    va, te = train_test_split(rest, test_size=0.50, stratify=rest_lab, random_state=seed)
    idx = lambda ps: np.where(np.isin(pid, ps))[0]
    return idx(tr), idx(va), idx(te)


def slice_split(y, seed):
    ids = np.arange(len(y))
    tr, rest = train_test_split(ids, test_size=0.30, stratify=y, random_state=seed)
    va, te = train_test_split(rest, test_size=0.50, stratify=y[rest], random_state=seed)
    return tr, va, te


class SliceDS(Dataset):
    def __init__(self, img, y, idx, train):
        self.img, self.y, self.idx = img, y, idx
        self.aug = T.Compose([
            T.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.1, contrast=0.1),
        ]) if train else None

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = self.idx[i]
        x = torch.from_numpy(self.img[j].astype(np.float32))[None, None]
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)[0].repeat(3, 1, 1)
        if self.aug is not None:
            x = self.aug(x).clamp(0, 1)
        return (x - MEAN) / STD, int(self.y[j])


# ------------------------------------------------------------------ models
class ChannelAttention(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(c, c // r, 1, bias=False), nn.ReLU(inplace=True),
                                 nn.Conv2d(c // r, c, 1, bias=False))

    def forward(self, x):
        a = self.mlp(F.adaptive_avg_pool2d(x, 1)) + self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(a)


class SpatialAttention(nn.Module):
    def __init__(self, k=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, k, padding=k // 2, bias=False)

    def forward(self, x):
        a = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], 1)
        return x * torch.sigmoid(self.conv(a))


def attention_block(c, channel, spatial):
    mods = ([ChannelAttention(c)] if channel else []) + ([SpatialAttention()] if spatial else [])
    return nn.Sequential(*mods)


class GatedAttention(nn.Module):
    """Identity-initialised attention: y = x + gamma * (A(x) - x), gamma learnable, initialised to 0.
    At the start of training the block is an identity, so ImageNet-pretrained features are preserved;
    the network learns how strongly to apply attention."""
    def __init__(self, c, channel, spatial):
        super().__init__()
        self.att = attention_block(c, channel, spatial)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x + self.gamma * (self.att(x) - x)


def build_model(name, pretrained):
    tv = torchvision.models
    if name.startswith("gated"):
        m = tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        ch = name in ("gated", "gated_slicesplit", "gated_channel")
        sp = name in ("gated", "gated_slicesplit", "gated_spatial")
        for lname, c in zip(["layer1", "layer2", "layer3", "layer4"], [256, 512, 1024, 2048]):
            setattr(m, lname, nn.Sequential(getattr(m, lname), GatedAttention(c, ch, sp)))
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, len(CLASSES)))
        return m, m.layer4
    if name in ("proposed", "proposed_slicesplit", "resnet50", "cbam_channel", "cbam_spatial"):
        m = tv.resnet50(weights=tv.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        ch = name in ("proposed", "proposed_slicesplit", "cbam_channel")
        sp = name in ("proposed", "proposed_slicesplit", "cbam_spatial")
        if ch or sp:
            for lname, c in zip(["layer1", "layer2", "layer3", "layer4"], [256, 512, 1024, 2048]):
                setattr(m, lname, nn.Sequential(getattr(m, lname), attention_block(c, ch, sp)))
        m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, len(CLASSES)))
        return m, m.layer4
    if name == "densenet121":
        m = tv.densenet121(weights=tv.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.classifier.in_features, len(CLASSES)))
        return m, None
    if name == "efficientnet_b0":
        m = tv.efficientnet_b0(weights=tv.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, len(CLASSES))
        return m, None
    if name == "vit_b_16":
        m = tv.vit_b_16(weights=tv.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        m.heads.head = nn.Linear(m.heads.head.in_features, len(CLASSES))
        return m, None
    raise ValueError(name)


# ------------------------------------------------------------------ train / predict
def train_model(model, tr_dl, va_dl, class_w, args, dev, log):
    # newly added attention modules (randomly initialised) learn 10x faster than the pretrained backbone
    att = [p for n, p in model.named_parameters() if ".att." in n or n.endswith(".gamma")]
    att_ids = {id(p) for p in att}
    rest = [p for p in model.parameters() if id(p) not in att_ids]
    opt = torch.optim.Adam([{"params": rest, "lr": 1e-4}] + ([{"params": att, "lr": 1e-3}] if att else []))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=5)
    crit = nn.CrossEntropyLoss(weight=class_w.to(dev))
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    best, best_state, bad = 1e9, None, 0
    for ep in range(args.epochs):
        model.train(); tl = 0
        for x, y in tr_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tl += loss.item() * len(y)
        model.eval(); vl = 0
        with torch.no_grad(), torch.autocast(dev.type, enabled=dev.type == "cuda"):
            for x, y in va_dl:
                x, y = x.to(dev), y.to(dev)
                vl += crit(model(x).float(), y).item() * len(y)
        tl /= len(tr_dl.dataset); vl /= len(va_dl.dataset)
        sched.step(vl)
        log(f"    epoch {ep + 1:3d}  train loss {tl:.4f}  val loss {vl:.4f}")
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                log(f"    early stop at epoch {ep + 1}")
                break
    model.load_state_dict(best_state)
    return model, ep + 1, best


@torch.no_grad()
def predict(model, dl, dev):
    model.eval(); P = []
    for x, _ in dl:
        with torch.autocast(dev.type, enabled=dev.type == "cuda"):
            P.append(torch.softmax(model(x.to(dev)).float(), 1).cpu().numpy())
    return np.concatenate(P)


# ------------------------------------------------------------------ metrics
def metrics(y, p):
    pred = p.argmax(1)
    cm = confusion_matrix(y, pred, labels=range(len(CLASSES)))
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, labels=range(len(CLASSES)), zero_division=0)
    spec = []
    for k in range(len(CLASSES)):
        tn = cm.sum() - cm[k].sum() - cm[:, k].sum() + cm[k, k]
        spec.append(tn / max(tn + cm[:, k].sum() - cm[k, k], 1))
    try:
        auc = roc_auc_score(y, p, multi_class="ovr", average="macro", labels=list(range(len(CLASSES))))
    except ValueError:
        auc = float("nan")
    return dict(accuracy=accuracy_score(y, pred), macro_f1=f1_score(y, pred, average="macro"),
                macro_auc=auc, precision=prec.tolist(), recall=rec.tolist(), specificity=spec,
                f1=f1.tolist(), support=cm.sum(1).tolist(), cm=cm.tolist())


def boot_ci(y, p, pid, n_boot, seed, other=None):
    """Patient-level bootstrap 95% CI for accuracy, macro F1, macro AUC.
    If `other` probabilities are given, also returns CI/P for the paired difference (self - other)."""
    rng = np.random.default_rng(seed)
    pats = np.unique(pid); groups = {q: np.where(pid == q)[0] for q in pats}
    keys = ["accuracy", "macro_f1", "macro_auc"]; S = defaultdict(list); D = defaultdict(list)
    for _ in range(n_boot):
        idx = np.concatenate([groups[q] for q in rng.choice(pats, len(pats), replace=True)])
        if len(np.unique(y[idx])) < len(CLASSES):
            continue
        a = metrics(y[idx], p[idx])
        for k in keys: S[k].append(a[k])
        if other is not None:
            b = metrics(y[idx], other[idx])
            for k in keys: D[k].append(a[k] - b[k])
    ci = {k: (float(np.nanpercentile(S[k], 2.5)), float(np.nanpercentile(S[k], 97.5))) for k in keys}
    diff = None
    if other is not None:
        diff = {k: dict(ci=(float(np.nanpercentile(D[k], 2.5)), float(np.nanpercentile(D[k], 97.5))),
                        p=float(min(1.0, 2 * min(np.mean(np.array(D[k]) <= 0), np.mean(np.array(D[k]) >= 0)))))
                for k in keys}
    return ci, diff


def mcnemar(y, pa, pb):
    a, b = pa.argmax(1) == y, pb.argmax(1) == y
    n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    return float(binomtest(n01, n01 + n10, 0.5).pvalue) if n01 + n10 > 0 else 1.0


def patient_level(y, p, pid):
    pats = np.unique(pid)
    return (np.array([np.bincount(y[pid == q]).argmax() for q in pats]),
            np.stack([p[pid == q].mean(0) for q in pats]))


# ------------------------------------------------------------------ Grad-CAM
class GradCAM:
    def __init__(self, model, layer):
        self.model, self.A, self.G = model, None, None
        def fwd(m, i, o):
            self.A = o
            if o.requires_grad:
                o.register_hook(lambda g: setattr(self, "G", g))
        layer.register_forward_hook(fwd)

    def __call__(self, x, cls):
        self.model.zero_grad(set_to_none=True)
        out = self.model(x)
        out[torch.arange(len(cls)), cls].sum().backward()
        w = self.G.mean((2, 3), keepdim=True)
        cam = F.relu((w * self.A).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=(512, 512), mode="bilinear", align_corners=False)[:, 0]
        cam = cam / (cam.flatten(1).amax(1).view(-1, 1, 1) + 1e-8)
        return cam.detach().cpu().numpy()


def gradcam_eval(model, layer, ds, img, mask, y, idx, p, dev, bs=16):
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(True)
    gc = GradCAM(model, layer); pred = p.argmax(1)
    rows, cams = [], {}
    for s in range(0, len(idx), bs):
        ii = list(range(s, min(s + bs, len(idx))))
        x = torch.stack([ds[i][0] for i in ii]).to(dev)
        cam = gc(x, torch.tensor(pred[ii], device=dev))
        for k, i in enumerate(ii):
            j = idx[i]; m = mask[j].astype(bool); c = cam[k]
            if m.sum() == 0:
                continue
            r, cc = np.unravel_index(np.argmax(c), c.shape)
            inter = np.logical_and(c >= 0.5, m).sum(); union = np.logical_or(c >= 0.5, m).sum()
            rows.append(dict(i=int(i), j=int(j), y=int(y[j]), correct=bool(pred[i] == y[j]), zero=bool(c.max() <= 1e-6),
                             hit=bool(m[r, cc]), energy=float(c[m].sum() / (c.sum() + 1e-8)),
                             iou=float(inter / max(union, 1))))
            cams[int(j)] = c.astype(np.float16)
    return rows, cams


def summarise_cam(rows):
    out = {}
    for k, name in list(enumerate(CLASSES)) + [(None, "All")]:
        r = [q for q in rows if q["correct"] and (k is None or q["y"] == k)]
        if not r:
            continue
        e = np.array([q["energy"] for q in r]); u = np.array([q["iou"] for q in r])
        out[name] = dict(n=len(r), hit=100 * np.mean([q["hit"] for q in r]), zero=100 * np.mean([q["zero"] for q in r]),
                         energy=(float(e.mean()), float(e.std())), iou=(float(u.mean()), float(u.std())))
    return out


# ------------------------------------------------------------------ figures
def fig_examples(img, mask, y, out):
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.3), dpi=200)
    for k, a in enumerate(ax):
        j = np.where(y == k)[0][0]
        a.imshow(img[j].astype(np.float32), cmap="gray"); a.contour(mask[j], levels=[0.5], colors="yellow", linewidths=1)
        a.set_title(f"({'abc'[k]}) {CLASSES[k]}", fontsize=10); a.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(out, "Figure1_examples.png")); plt.close(fig)


def fig_cm(cm, out):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=200)
    im = ax.imshow(cm, cmap="Blues"); fig.colorbar(im, fraction=.046, pad=.04)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(len(CLASSES)), CLASSES); ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    fig.tight_layout(); fig.savefig(os.path.join(out, "Figure3_confusion_matrix.png")); plt.close(fig)


def fig_roc(y, p, out):
    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=200)
    for k, c in enumerate(CLASSES):
        fpr, tpr, _ = roc_curve(y == k, p[:, k])
        ax.plot(fpr, tpr, label=f"{c} (AUC = {roc_auc_score(y == k, p[:, k]):.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(frameon=False, loc="lower right"); fig.tight_layout()
    fig.savefig(os.path.join(out, "Figure4_ROC.png")); plt.close(fig)


def fig_gradcam(img, mask, rows, cams, out):
    picks = []
    for k in range(len(CLASSES)):
        r = sorted([q for q in rows if q["correct"] and q["y"] == k], key=lambda q: -q["energy"])
        if r: picks.append((r[len(r) // 4], f"{CLASSES[k]} (correct)"))
    bad = sorted([q for q in rows if not q["hit"]], key=lambda q: q["energy"])
    if bad: picks.append((bad[0], f"{CLASSES[bad[0]['y']]} (failure case)"))
    fig, ax = plt.subplots(len(picks), 2, figsize=(6, 3 * len(picks)), dpi=200, squeeze=False)
    for r, (q, t) in enumerate(picks):
        im = img[q["j"]].astype(np.float32)
        ax[r, 0].imshow(im, cmap="gray"); ax[r, 0].set_title(t, fontsize=9)
        ax[r, 1].imshow(im, cmap="gray"); ax[r, 1].imshow(cams[q["j"]].astype(np.float32), cmap="jet", alpha=0.4)
        ax[r, 1].contour(mask[q["j"]], levels=[0.5], colors="white", linewidths=1)
        ax[r, 1].set_title("Grad-CAM + expert mask", fontsize=9)
        for a in ax[r]: a.axis("off")
    fig.tight_layout(); fig.savefig(os.path.join(out, "Figure5_gradcam.png")); plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/figshare_brain_tumour",
                    help="folder with the .mat files; downloaded automatically if empty")
    ap.add_argument("--out", default="results")
    ap.add_argument("--runs", nargs="+", default=RUNS_ALL)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no_pretrained", action="store_true", help="for testing without internet only")
    ap.add_argument("--ref", default="proposed", help="reference model for P values, confusion matrix, ROC, Grad-CAM figure")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logf = open(os.path.join(args.out, "train_log.txt"), "a")
    def log(s):
        print(s); logf.write(s + "\n"); logf.flush()
    log(f"Device: {dev} | torch {torch.__version__} | torchvision {torchvision.__version__}")

    if not os.path.exists(os.path.join(args.out, "cache.npz")) and count_mat(args.data_dir) == 0:
        download_dataset(args.data_dir)
    img, mask, y, pid = load_dataset(args.data_dir, os.path.join(args.out, "cache.npz"))
    log(f"Dataset: {len(y)} slices, {len(np.unique(pid))} patients")
    data_info = {c: dict(patients=int(len(np.unique(pid[y == k]))), slices=int((y == k).sum()))
                 for k, c in enumerate(CLASSES)}
    splits = {"patient": patient_split(y, pid, args.split_seed), "slice": slice_split(y, args.split_seed)}
    tr, va, te = splits["patient"]
    assert not set(pid[tr]) & set(pid[te]) and not set(pid[tr]) & set(pid[va]) and not set(pid[va]) & set(pid[te])
    split_info = {n: dict(patients=int(len(np.unique(pid[s]))), slices=int(len(s)))
                  for n, s in zip(["train", "val", "test"], splits["patient"])}
    log(f"Patient-level split: {split_info}")
    fig_examples(img, mask, y, args.out)

    R = {}
    for run in args.runs:
        split = splits["slice" if run.endswith("slicesplit") else "patient"]
        tr, va, te = split
        cw = torch.tensor(len(tr) / (len(CLASSES) * np.bincount(y[tr], minlength=len(CLASSES))), dtype=torch.float32)
        ds = {n: SliceDS(img, y, s, n == "train") for n, s in zip(["train", "val", "test"], split)}
        dl = {n: DataLoader(ds[n], batch_size=args.batch, shuffle=n == "train", num_workers=args.workers,
                            pin_memory=dev.type == "cuda") for n in ds}
        R[run] = dict(seeds={})
        for seed in args.seeds:
            done_file = os.path.join(args.out, f"run_{run}_seed{seed}.json")
            if os.path.exists(done_file) and os.path.exists(os.path.join(args.out, f"probs_{run}_seed{seed}.npy")):
                R[run]["seeds"][seed] = json.load(open(done_file))
                log(f"\n=== {run} | seed {seed} === already finished, skipping (delete {done_file} to redo)")
                continue
            log(f"\n=== {run} | seed {seed} ===")
            torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
            model, layer = build_model(run, not args.no_pretrained)
            model = model.to(dev)
            t0 = time.time()
            model, n_ep, best_vl = train_model(model, dl["train"], dl["val"], cw, args, dev, log)
            pv = predict(model, dl["val"], dev)
            val_acc = float((pv.argmax(1) == y[va]).mean())
            log(f"    best val loss {best_vl:.4f}  val acc {val_acc:.4f}")
            p = predict(model, dl["test"], dev)
            # inference time per slice
            x = torch.randn(1, 3, 224, 224, device=dev); model.eval()
            with torch.no_grad():
                for _ in range(5): model(x)
                if dev.type == "cuda": torch.cuda.synchronize()
                t1 = time.time()
                for _ in range(50): model(x)
                if dev.type == "cuda": torch.cuda.synchronize()
            ms = (time.time() - t1) / 50 * 1000
            m = metrics(y[te], p)
            yp, pp = patient_level(y[te], p, pid[te])
            m_pat = metrics(yp, pp)
            log(f"    test acc {m['accuracy']:.4f}  macroF1 {m['macro_f1']:.4f}  AUC {m['macro_auc']:.4f}  "
                f"(patient-level acc {m_pat['accuracy']:.4f})  epochs {n_ep}  train time {time.time() - t0:.0f}s")
            entry = dict(metrics=m, patient_metrics=m_pat, epochs=n_ep, infer_ms=ms,
                         val_loss=float(best_vl), val_acc=val_acc,
                         params_m=sum(q.numel() for q in model.parameters() if q.requires_grad) / 1e6)
            np.save(os.path.join(args.out, f"probs_{run}_seed{seed}.npy"), p)
            gam = [float(mm.gamma.detach().cpu()) for mm in model.modules() if isinstance(mm, GatedAttention)]
            if gam:
                entry["gamma"] = gam; log(f"    learned attention gates (layer1-4): {[round(g, 3) for g in gam]}")
            if seed == args.seeds[0] and layer is not None and run in ("proposed", "resnet50", "gated"):
                log("    computing Grad-CAM ...")
                rows, cams = gradcam_eval(model, layer, ds["test"], img, mask, y, te, p, dev)
                entry["gradcam"] = summarise_cam(rows)
                if run == args.ref:
                    fig_gradcam(img, mask, rows, cams, args.out)
            R[run]["seeds"][seed] = entry
            json.dump(entry, open(done_file, "w"), default=float)
            del model; torch.cuda.empty_cache() if dev.type == "cuda" else None

    # ---- statistics on first seed
    s0 = args.seeds[0]
    load = lambda r: np.load(os.path.join(args.out, f"probs_{r}_seed{s0}.npy"))
    te = splits["patient"][2]
    ref = args.ref
    if ref in R:
        P0 = load(ref)
        fig_cm(R[ref]["seeds"][s0]["metrics"]["cm"], args.out)
        fig_roc(y[te], P0, args.out)
    for run in R:
        tt = splits["slice" if run.endswith("slicesplit") else "patient"][2]
        P = load(run)
        other = P0 if (ref in R and run != ref and not run.endswith("slicesplit")) else None
        ci, diff = boot_ci(y[tt], P, pid[tt], args.n_boot, 0, other)
        R[run]["ci"] = ci
        if other is not None:
            R[run]["vs_proposed"] = dict(mcnemar_p=mcnemar(y[tt], P0, P),
                                         bootstrap=boot_ci(y[tt], P0, pid[tt], args.n_boot, 0, P)[1])
        accs = [R[run]["seeds"][s]["metrics"]["accuracy"] for s in args.seeds]
        R[run]["acc_mean_sd"] = (float(np.mean(accs)), float(np.std(accs)))

    json.dump(dict(data=data_info, split=split_info, args=vars(args), runs=R),
              open(os.path.join(args.out, "results.json"), "w"), indent=1, default=float)
    write_summary(R, data_info, split_info, args, os.path.join(args.out, "results_summary.txt"))
    log(f"\nDone. Send {os.path.join(args.out, 'results_summary.txt')} and the Figure*.png files.")


def write_summary(R, data_info, split_info, args, path):
    s0 = args.seeds[0]; L = []
    pc = lambda v: f"{100 * v:.1f}"
    ci = lambda c: f"({pc(c[0])}–{pc(c[1])})"
    fp = lambda p, nb=None: (f"<{1 / nb:.4f}" if nb and p == 0 else f"{p:.4f}")
    L.append("RESULTS SUMMARY — brain tumour classification\n")
    L.append(f"Seeds: {args.seeds}   bootstrap: {args.n_boot}   split seed: {args.split_seed}\n")
    L.append("Dataset:"); [L.append(f"  {k}: {v['patients']} patients, {v['slices']} slices") for k, v in data_info.items()]
    L.append("Patient-level split:"); [L.append(f"  {k}: {v['patients']} patients, {v['slices']} slices") for k, v in split_info.items()]
    L.append("\nTABLE 4/6/7 — model comparison (test set, slice level; 95% CI by patient bootstrap)")
    L.append("run | acc (CI) | macroF1 (CI) | macroAUC (CI) | acc mean±SD over seeds | patient-level acc | params M | ms/slice | epochs | best val loss | val acc | McNemar P vs " + args.ref + " | AUC diff P")
    for run, r in R.items():
        e = r["seeds"][s0]; m = e["metrics"]; c = r["ci"]; vs = r.get("vs_proposed")
        L.append(f"{run} | {pc(m['accuracy'])} {ci(c['accuracy'])} | {pc(m['macro_f1'])} {ci(c['macro_f1'])} | "
                 f"{m['macro_auc']:.3f} ({c['macro_auc'][0]:.3f}–{c['macro_auc'][1]:.3f}) | "
                 f"{pc(r['acc_mean_sd'][0])} ± {pc(r['acc_mean_sd'][1])} | {pc(e['patient_metrics']['accuracy'])} | "
                 f"{e['params_m']:.1f} | {e['infer_ms']:.1f} | {e['epochs']} | {e.get('val_loss', float('nan')):.4f} | {pc(e.get('val_acc', float('nan')))} | "
                 f"{fp(vs['mcnemar_p'])} | {fp(vs['bootstrap']['macro_auc']['p'], args.n_boot)}" if vs else
                 f"{run} | {pc(m['accuracy'])} {ci(c['accuracy'])} | {pc(m['macro_f1'])} {ci(c['macro_f1'])} | "
                 f"{m['macro_auc']:.3f} ({c['macro_auc'][0]:.3f}–{c['macro_auc'][1]:.3f}) | "
                 f"{pc(r['acc_mean_sd'][0])} ± {pc(r['acc_mean_sd'][1])} | {pc(e['patient_metrics']['accuracy'])} | "
                 f"{e['params_m']:.1f} | {e['infer_ms']:.1f} | {e['epochs']} | {e.get('val_loss', float('nan')):.4f} | {pc(e.get('val_acc', float('nan')))} | - | -")
    if args.ref in R:
        m = R[args.ref]["seeds"][s0]["metrics"]
        L.append(f"\nTABLE 5 — per-class ({args.ref})")
        L.append("class | precision | recall | specificity | F1 | n")
        for k, c in enumerate(CLASSES):
            L.append(f"{c} | {pc(m['precision'][k])} | {pc(m['recall'][k])} | {pc(m['specificity'][k])} | {pc(m['f1'][k])} | {m['support'][k]}")
        L.append(f"Macro | {pc(np.mean(m['precision']))} | {pc(np.mean(m['recall']))} | {pc(np.mean(m['specificity']))} | {pc(m['macro_f1'])} | {sum(m['support'])}")
        L.append("Confusion matrix (rows true, cols predicted; " + ", ".join(CLASSES) + "):")
        for row in m["cm"]: L.append("  " + "  ".join(str(v) for v in row))
    L.append("\nLearned attention gates gamma (layer1..layer4), seed " + str(s0) + ":")
    for run, r in R.items():
        g = r["seeds"][s0].get("gamma")
        if g: L.append(f"  {run}: " + ", ".join(f"{v:.3f}" for v in g))
    L.append("\nTABLE 8 — Grad-CAM vs expert masks (correctly classified test slices)")
    L.append("model | class | n | pointing-game hit % | energy ratio mean±SD | IoU mean±SD | empty maps %")
    for run in ("proposed", "gated", "resnet50"):
        g = R.get(run, {}).get("seeds", {}).get(s0, {}).get("gradcam")
        if g:
            for c, v in g.items():
                L.append(f"{run} | {c} | {v['n']} | {v['hit']:.1f} | {v['energy'][0]:.3f} ± {v['energy'][1]:.3f} | {v['iou'][0]:.3f} ± {v['iou'][1]:.3f} | {v.get('zero', float('nan')):.1f}")
    open(path, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
