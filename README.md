# Patient-Aware Melanoma Classification (EEEM068 LSA)

A controlled study of evaluation protocol, class balancing and clinical metadata
fusion for melanoma classification on SIIM-ISIC 2020.

**Research question:** does patient-independent evaluation change what we conclude
about melanoma classifiers, and does clinical metadata add complementary value
under that stricter protocol?

## Key findings

- The image-level split used in prior work places 1,704 of 2,056 patients (82.9%)
  in both training and validation partitions.
- Correcting the split did not reduce measured performance (0.8975 leaky vs
  0.9034 patient-disjoint), contrary to the initial hypothesis.
- Run-to-run variance from initialisation alone (sd 0.007 ROC-AUC) exceeds every
  difference between configurations tested.
- Class balancing improves recall at a fixed threshold far more (0.000 -> 0.427)
  than at a matched 10% FPR operating point (0.615 -> 0.735), indicating most of
  the apparent gain is threshold relocation rather than better discrimination.
- At matched FPR the model is less sensitive for female patients (0.652 vs 0.732)
  and for ages 40-55 (0.576 vs 0.857 for over-70s).

## Features

- Dataset integrity audit with leakage assertion (`src/audit_split.py`)
- Two split protocols: patient-disjoint (StratifiedGroupKFold) and image-level,
  matched on positive count for controlled comparison
- Metadata encoder with explicit missingness handling, fitted on training rows only
- Configurable late-fusion ViT: image-only or image+metadata from one class
- Ablation by field masking (dimensionality held constant)
- Bootstrap confidence intervals on all metrics
- Matched-operating-point evaluation (recall at fixed FPR)
- Seed control for variance quantification
- Per-epoch JSONL training logs for every run

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/audit_split.py --csv <path>/train.csv --img-dir <path>/train

python -m src.train --name E0_baseline --csv <path>/train.csv --img-dir <path>/train
python -m src.train --name E1_previous_wrs --balanced --csv ... --img-dir ...
python -m src.train --name E2_multimodal --balanced --use-metadata --csv ... --img-dir ...
```

Analysis and figures: `Melanoma-Classification-Karthik.ipynb`

## Layout

| Path | Purpose |
|---|---|
| `src/audit_split.py` | Dataset audit, split generation, leakage verification |
| `src/dataset.py` | Transforms, metadata encoding, dataloaders, balanced sampling |
| `src/models.py` | ViT-B/16 with optional metadata fusion branch |
| `src/metrics.py` | Metric suite and bootstrap confidence intervals |
| `src/train.py` | Single configurable training entry point for all experiments |
| `logs/` | Per-epoch training logs and observations |
| `results/` | Predictions, configs, tables, figures |
| `splits/` | Fixed split assignments (seed 42) |

Data is not committed. SIIM-ISIC 2020, 512x512 JPEG release.
