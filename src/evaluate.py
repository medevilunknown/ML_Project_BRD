"""Metrics and group-aware cross-validation helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]


def score_predictions(y_true, y_pred, y_probability=None) -> dict[str, float]:
    scores = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": np.nan,
    }
    if y_probability is not None and len(np.unique(y_true)) > 1:
        scores["ROC-AUC"] = roc_auc_score(y_true, y_probability)
    return scores


def evaluate_models(models, X_test, y_test, thresholds=None, extra_probabilities=None) -> tuple[pd.DataFrame, dict]:
    """Score each model on the test calves at the default 0.5 and at its tuned threshold.

    `extra_probabilities` holds already-computed test probabilities of models that do not
    take the tabular X_test (the 1D-CNN uses history windows)."""
    thresholds = thresholds or {}
    probs = {name: model.predict_proba(X_test)[:, 1] for name, model in models.items()}
    probs.update(extra_probabilities or {})
    rows, predictions = [], {}
    for name, prob in probs.items():
        t = thresholds.get(name, 0.5)
        pred = (prob >= t).astype(int)
        default = score_predictions(y_test, (prob >= 0.5).astype(int), prob)
        rows.append({"Model": name, "Threshold": t, **score_predictions(y_test, pred, prob),
                     "Recall@0.5": default["Recall"], "F1@0.5": default["F1"]})
        predictions[name] = {"pred": pred, "prob": prob}
    return pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True), predictions


def grouped_cross_validate(models, X, y, groups, thresholds=None, n_splits=5, random_state=42) -> pd.DataFrame:
    """Repeat the evaluation over 5 calf-grouped folds to measure stability (mean +/- std)."""
    thresholds = thresholds or {}
    try:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        splits = splitter.split(X, y, groups)
    except (ImportError, ValueError):
        splitter = GroupKFold(n_splits=n_splits)
        splits = splitter.split(X, y, groups)
    split_list = list(splits)
    rows = []
    for name, model in models.items():
        fold_scores = []
        for train_idx, valid_idx in split_list:
            candidate = clone(model)
            candidate.fit(X.iloc[train_idx], y.iloc[train_idx])
            prob = candidate.predict_proba(X.iloc[valid_idx])[:, 1]
            pred = (prob >= thresholds.get(name, 0.5)).astype(int)
            fold_scores.append(score_predictions(y.iloc[valid_idx], pred, prob))
        row = {"Model": name}
        for metric in METRIC_NAMES:
            values = [s[metric] for s in fold_scores if not np.isnan(s[metric])]
            row[f"{metric} mean"] = np.mean(values) if values else np.nan
            row[f"{metric} std"] = np.std(values) if values else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def plot_learning_curve(estimator, X, y, groups, path, title, random_state=42):
    """Training vs validation score as the number of training records grows."""
    import matplotlib.pyplot as plt
    from sklearn.model_selection import learning_curve
    sizes, train_scores, valid_scores = learning_curve(
        estimator, X, y, groups=groups, cv=StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state),
        train_sizes=np.linspace(0.15, 1.0, 6), scoring="roc_auc", n_jobs=-1)
    plt.figure(figsize=(7, 5))
    for scores, label, colour in [(train_scores, "Training", "#3b6378"), (valid_scores, "Validation (unseen calves)", "#c95c4b")]:
        mean, std = scores.mean(1), scores.std(1)
        plt.plot(sizes, mean, marker="o", label=label, color=colour); plt.fill_between(sizes, mean - std, mean + std, alpha=.15, color=colour)
    plt.title(title); plt.xlabel("Training records"); plt.ylabel("ROC-AUC"); plt.ylim(0.4, 1.02); plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=180); plt.close()
    return {"train_auc_final": float(train_scores.mean(1)[-1]), "valid_auc_final": float(valid_scores.mean(1)[-1])}


def plot_boosting_loss(pipeline, X_train, y_train, groups, path, random_state=42):
    """Log-loss per boosting iteration on training calves vs a held-out group of calves."""
    import matplotlib.pyplot as plt
    from sklearn.base import clone
    from sklearn.metrics import log_loss
    from sklearn.model_selection import GroupShuffleSplit
    tr, va = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state).split(X_train, y_train, groups))
    model = clone(pipeline).fit(X_train.iloc[tr], y_train.iloc[tr])
    prep, gb = model.named_steps["preprocess"], model.named_steps["model"]
    Xt, Xv = prep.transform(X_train.iloc[tr]), prep.transform(X_train.iloc[va])
    train_loss = [log_loss(y_train.iloc[tr], p[:, 1], labels=[0, 1]) for p in gb.staged_predict_proba(Xt)]
    valid_loss = [log_loss(y_train.iloc[va], p[:, 1], labels=[0, 1]) for p in gb.staged_predict_proba(Xv)]
    plt.figure(figsize=(7, 5))
    plt.plot(train_loss, label="Training loss", color="#3b6378"); plt.plot(valid_loss, label="Validation loss (unseen calves)", color="#c95c4b")
    best = int(np.argmin(valid_loss)); plt.axvline(best, ls="--", c="grey", label=f"Lowest validation loss (iter {best + 1})")
    plt.title("Gradient Boosting: log-loss vs boosting iterations"); plt.xlabel("Boosting iteration"); plt.ylabel("Log-loss"); plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=180); plt.close()
    return {"best_iteration": best + 1, "final_train_loss": train_loss[-1], "final_valid_loss": valid_loss[-1], "min_valid_loss": min(valid_loss)}
