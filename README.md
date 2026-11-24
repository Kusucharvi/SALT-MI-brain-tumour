# SALT-MI: Evaluating Saliency Maps for Deep Learning-Based Brain Tumour Classification on MRI

Code for the paper *"Evaluating Saliency Maps for Deep Learning-Based Brain Tumour Classification on MRI: The SALT-MI Protocol"* ({{AUTHORS}}, {{JOURNAL, YEAR}}).

Every number, table and figure in the paper is produced by the scripts below. All datasets are public and are downloaded automatically.

## Data

| Dataset | Use in the paper | Source |
|---|---|---|
| Cheng et al. brain tumour dataset (3,064 contrast-enhanced T1-weighted slices, 233 patients, expert masks) | Training and all main analyses | figshare, doi:10.6084/m9.figshare.1512427 |
| BRISC 2025 | Duplicate screening | Zenodo record 17524350 |
| Medical Segmentation Decathlon, Task01 BrainTumour (BraTS-derived gliomas) | Independent external validation only (never used for training) | AWS Open Data (`msd-for-monai`) |

## Requirements

Python ≥ 3.10 and a CUDA GPU (all experiments were run on one NVIDIA Tesla T4 in Google Colab).

```bash
pip install -r requirements.txt
```

## Scripts

| Script | Paper content | Approx. time (T4) |
|---|---|---|
| `run_experiments.py` | Shared code (data download and loading, models, training). Run alone: single-split benchmark, attention ablation, leakage (Suppl. S1) | ~1.5 h |
| `run_multiplicity.py` | Main study: 2 architectures × 5 patient-wise folds × 5 seeds; Grad-CAM, Grad-CAM++, HiResCAM; baselines; randomisation test; ensembles; consensus regions (Sections 4.1–4.5) | ~4.5 h |
| `run_occlusion.py` | Occlusion vs. Grad-CAM on the same ResNet-50 models (Section 4.6) | ~2.5 h |
| `run_external.py` | BRISC download, duplicate screening, models trained on all primary data (Section 4.7, Suppl. S3) | ~2 h |
| `verify_duplicates.py` | Visual and label check of duplicate detection (Suppl. Fig. S3) | ~2 min |
| `run_leakage_fingerprint.py` | Leakage fingerprint (Suppl. S2) | ~3 h |
| `run_guided.py` | Mask-guided training vs. unguided, same folds and seeds (Section 4.8) | ~3 h |
| `run_external_msd.py` | Independent validation on MSD Task01 (Section 4.9, Suppl. S4) | ~1–1.5 h |
| `run_guided_msd.py` | Mask-guided models on MSD (Section 4.10) | ~1–1.5 h |

## Reproduce (run in this order)

```bash
# 1. main study (downloads the primary dataset on first run)
python run_multiplicity.py --data_dir data/figshare --out results_mult2 --archs resnet50 densenet121
# 2. occlusion
python run_occlusion.py --out results_occ --cache results_mult2/cache.npz
# 3. BRISC duplicate screening
python run_external.py --out results_ext --cache results_mult2/cache.npz
python verify_duplicates.py --cache results_mult2/cache.npz --brisc_cache results_ext/brisc_cache.npz --out results_ext
# 4. mask-guided training
python run_guided.py --cache results_mult2/cache.npz --out results_guided --baseline results_occ
# 5. independent validation on MSD (uses the models saved in step 3)
python run_external_msd.py --models results_ext/models --cache results_mult2/cache.npz --out results_msd
# 6. mask-guided models on MSD
python run_guided_msd.py --cache results_mult2/cache.npz --msd results_msd --out results_msd_guided
# optional: supplementary analyses
python run_leakage_fingerprint.py --data_dir data/figshare --out results_mult --cache results_mult2/cache.npz
python run_experiments.py --out results_v2 --ref gated
```

All scripts are resumable: if a run is interrupted, re-run the same command and finished models are skipped. Each script writes a `*_summary.txt` with the numbers reported in the paper.

## The SALT-MI protocol

1. **Localise against masks:** score every test map against expert lesion masks.
2. **Compare with trivial baselines:** a random point and the image centre.
3. **Vary the training seed:** at least five models; report hit flips and hits by all seeds.
4. **Use more than one explanation method:** gradient-based and perturbation-based, plus a randomisation test.
5. **Test on independent, screened data:** screen for near-duplicates, then repeat steps 1–4.

## Determinism

Patient-wise folds use `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`; training seeds are 0–4. Re-training with the same seeds reproduced all 25 ResNet-50 test accuracies exactly on the same GPU type; small differences may occur on other hardware.

## Citation

```
{{Paper citation and Zenodo DOI, added after publication}}
```

## Licence

MIT (see `LICENSE`). The datasets are subject to their own licences.
