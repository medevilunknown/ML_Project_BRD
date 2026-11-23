# Early Detection of Bovine Respiratory Disease (BRD) in Dairy Calves with Machine Learning

**ML Course Project, Group 19.** Team: Dana Ram Patel (lead), Prajwal M.A, Rohit Kumar Gond

Can automatic-feeder and activity-sensor data flag a calf **1–6 days before** it is clinically diagnosed with BRD (pneumonia)?
We compare five classical/tabular models, an MLP and a 1D-CNN, add an unsupervised PCA + K-Means analysis, and run a detailed failure-case analysis.

## Headline results (22 unseen test calves, F2-tuned thresholds)

| Model | Recall | Precision | F1 | ROC-AUC | ROC-AUC, age-matched* |
|---|---|---|---|---|---|
| **Random Forest (selected)** | **0.82** | **0.16** | **0.26** | **0.78** | **0.59** |
| Gradient Boosting | 0.65 | 0.14 | 0.23 | 0.70 | 0.44 |
| MLP | 0.61 | 0.13 | 0.21 | 0.70 | 0.50 |
| KNN | 0.70 | 0.11 | 0.20 | 0.64 | 0.52 |
| Logistic Regression | 0.97 | 0.11 | 0.19 | 0.68 | 0.48 |
| 1D-CNN (7-day window) | 0.61 | 0.11 | 0.19 | 0.69 | 0.49 |

\*Only records from calves ≤ 42 days on the feeder, the age range in which every BRD case occurred.
The gap between the last two columns is the main finding of the failure analysis: much of the apparent skill comes from calf **age**.
All 11 sick test calves were flagged on at least one of their 6 pre-diagnosis days, but all 11 healthy test calves also received false alarms.
Full numbers are in `outputs/results/`.

## How to run

Requires Python 3.10+ (tested on 3.12, macOS arm64, CPU only).

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py                     # full pipeline, ~3–4 min on a laptop CPU
```

`python main.py` regenerates every figure in `outputs/figures/`, every table in `outputs/results/`, and the saved models in `outputs/models/`.

Other entry points:

```bash
jupyter notebook notebooks/brd_analysis.ipynb   # step-by-step walkthrough (already executed, outputs saved)
streamlit run app.py                            # interactive demo using the saved model
```

## Dataset

`data/Cantor_machineclean-opendepository.xlsx` is included in this repository. It is the open-access calf dataset of Cantor et al. (Univ. of Kentucky / Penn State),
with 8,706 daily records, 36 columns and 107 calves, from automatic milk feeders, pedometers and twice-weekly health scoring.

- **Positive (1):** records with `-6 ≤ Pneuday ≤ -1` (the 6 days before BRD diagnosis), giving 316 records from 54 calves.
- **Negative (0):** every record of the 57 calves that were never diagnosed, giving 4,298 records.
- **Features (26):** milk/starter intake, drinking speed, rewarded/unrewarded visits, steps, lying time and bouts, temperature, health scores, weight, age, etc.
  Identifiers, dates, `Pneuday`, `Count` (row counter), `Observation` (index), `agefeeder` (duplicate) and the constant `Herd` are excluded.

## Methodology in brief

1. **Cleaning:** `Lying_time` (HH:MM:SS) is converted to minutes, scorer-name spellings are merged, and leakage/ID columns are removed.
2. **Unsupervised:** median-impute and standardise, then PCA (keeping 90% of variance, 16 components), then K-Means with k = 2–8, chosen by silhouette score.
3. **Split:** 80/20 `GroupShuffleSplit` by calf (85 train / 22 test calves). No calf appears in both sets.
4. **Tuning:** `GridSearchCV` with 5-fold `StratifiedGroupKFold` on training calves, scored by average precision.
5. **Threshold:** the F2-optimal probability cut-off is chosen from out-of-fold training predictions. F2 is used because missing a sick calf costs more than a false alarm.
6. **Models:** Logistic Regression (L1/L2), KNN, Random Forest, Gradient Boosting, MLP (sklearn), and a 1D-CNN (PyTorch, two Conv1d layers over a 7-day × 26-channel window, weighted BCE loss, early stopping).
7. **Selection:** the model with the best training-CV average precision is chosen. The test set is never used for any decision.
8. **Evaluation:** accuracy, precision, recall, F1 and ROC-AUC on test calves, plus 5-fold grouped CV (mean ± std), learning/loss curves, and the failure analysis.

## Code structure

| File | Purpose |
|---|---|
| `main.py` | Entry point; calls `src.train.run_project()` |
| `src/data_preprocessing.py` | Loading, column detection, cleaning (`clean_columns`), target construction, feature selection, preprocessing pipeline, calf-level split |
| `src/clustering.py` | PCA + K-Means, elbow/silhouette selection, cluster profiles and plots |
| `src/tuning.py` | Model definitions and hyperparameter grids (incl. MLP), grouped grid search, F2 threshold tuning |
| `src/deep_models.py` | 1D-CNN: 7-day history windows, PyTorch model, training with early stopping, grouped CV |
| `src/evaluate.py` | Metrics, grouped cross-validation, learning curve, boosting loss curve |
| `src/failure_analysis.py` | Probability overlap, error feature profiles, errors by age, age-matched AUC, per-calf and per-model analysis |
| `src/train.py` | Orchestrates the full experiment, saves figures, results and models |
| `src/utils.py` | Paths |
| `app.py` | Streamlit demo |
| `notebooks/brd_analysis.ipynb` | Executed walkthrough |

## Outputs

- `outputs/models/best_brd_model.joblib`: selected Random Forest pipeline, tuned threshold and feature lists. Load it with `joblib.load` and use `artifact["pipeline"].predict_proba(X)[:, 1] >= artifact["threshold"]`.
- `outputs/models/cnn_1d.pt`: 1D-CNN weights (PyTorch `state_dict`).
- `outputs/results/`: `model_results.csv`, `group_cross_validation.csv`, `hyperparameter_tuning.csv`, `clustering_scores.csv`, `cluster_profiles.csv`, `failure_*.csv`, `age_matched_auc.csv`, `recall_by_day.csv`, `run_summary.json`.
- `outputs/figures/`: about 20 plots (ROC, confusion matrices, feature importance, learning/loss curves, clustering, failure analysis).

## Disclaimer

This is an academic research prototype, not a veterinary diagnostic tool. Model associations are not causal.
