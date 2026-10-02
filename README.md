Repo for Dean Yao's Summer Undergraduate Research Fellowship (SURF) work at the University of Pennsylvania.

Code for my 2025 SURF at the University of Pennsylvania with Prof. Pratik Chaudhari.

**Goal:** Predict brain age and diagnosis (CN / MCI / AD) at 24, 48, 72, 96, and 120 months after a patient's baseline MRI.

**Cohort:** ADNI patients who were cognitively normal at baseline and later converted to MCI or AD. That is 114 patients and 840 scans, split 80/20 by patient so no patient appears in both train and test.

## Pipeline

| Step | Files | What it does |
|---|---|---|
| 1. Stage One: cross-sectional model | `final_model.py`, `final_model_inf.py` | Filters the cohort and makes the patient-level split. Runs 5-fold GroupKFold CV and trains an AutoGluon regressor that predicts months since baseline from a single scan's MUSE brain volumes. The `_inf` script evaluates on the held-out test set and plots the results. |
| 2. Build checkpoint targets | `label.py` | Predicts brain age for every scan. Training patients get out-of-fold predictions from models that never saw them. Each patient's trajectory is interpolated or extrapolated to every checkpoint, and diagnosis is taken from the nearest scan. Every target is flagged as interpolated or extrapolated. Writes one baseline row per patient to `train_val.parquet` / `test.parquet`. |
| 3. Stage Two: checkpoint models | `train24.py` … `train120.py` | For each checkpoint, trains a diagnosis classifier (macro-F1) and a brain-age regressor (RMSE). All columns from later checkpoints are dropped first. |
| 4. Evaluation | `eval24.py` … `eval120.py`, `diag_eval.py` | Scatter and residual plots for brain age, with extrapolated targets marked. `diag_eval.py` plots per-class F1 at each checkpoint, with the extrapolated share hatched. The `_remout` variants are the outlier-removed versions reported in the paper's footnotes. |

`brain_age_months.py` / `brain_age_months_inf.py` are an earlier iteration of Stage One.
