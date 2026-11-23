"""Hyperparameter search and decision-threshold tuning.

All searches use StratifiedGroupKFold on the TRAINING calves only, so every
validation fold contains calves the model has never seen (no calf-level leakage)
and the held-out test calves are never used for any choice.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# average_precision (area under the precision-recall curve) is used for model
# selection because it is threshold-independent and, unlike accuracy or ROC-AUC,
# focuses on how well the rare positive (pre-BRD) class is ranked.
SEARCH_SCORING = "average_precision"


def model_grids(random_state: int = 42) -> dict:
    """Base estimator and hyperparameter grid for each model."""
    return {
        "Logistic Regression": (
            LogisticRegression(solver="liblinear", class_weight="balanced", max_iter=5000, random_state=random_state),
            {"model__C": [0.01, 0.1, 1.0, 10.0], "model__penalty": ["l1", "l2"]},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {"model__n_neighbors": [5, 15, 31, 51], "model__weights": ["uniform", "distance"]},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=random_state, n_jobs=-1),
            {"model__max_depth": [4, 8, None], "model__min_samples_leaf": [1, 5, 20]},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=random_state),
            {"model__n_estimators": [100, 200], "model__learning_rate": [0.05, 0.1], "model__max_depth": [2, 3]},
        ),
        # Multi-layer perceptron: ReLU hidden layers, Adam optimiser, L2 penalty `alpha`,
        # early stopping on an internal 10% validation split.
        "MLP": (
            MLPClassifier(max_iter=500, early_stopping=True, validation_fraction=0.1, n_iter_no_change=15, random_state=random_state),
            {"model__hidden_layer_sizes": [(32,), (64, 32)], "model__alpha": [1e-3, 1e-1, 1.0], "model__learning_rate_init": [1e-3]},
        ),
    }


def group_cv(n_splits: int = 5, random_state: int = 42) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def tune_model(pipeline, grid, X, y, groups, random_state: int = 42) -> GridSearchCV:
    """Grid-search one pipeline with calf-grouped cross-validation."""
    search = GridSearchCV(pipeline, grid, scoring=SEARCH_SCORING, cv=group_cv(random_state=random_state), n_jobs=-1, refit=True)
    search.fit(X, y, groups=groups)
    return search


def tune_threshold(estimator, X, y, groups, random_state: int = 42) -> tuple[float, dict]:
    """Choose the probability cut-off that maximises F2 on out-of-fold predictions.

    F2 = 5PR / (4P + R) weights recall twice as much as precision, because missing a
    calf that is about to develop BRD costs more than an extra health check.

    With only ~7% positive rows, the default 0.5 cut-off makes tree models predict
    almost everything as healthy. Moving the threshold trades precision for recall.
    """
    oof = cross_val_predict(estimator, X, y, groups=groups, cv=group_cv(random_state=random_state), method="predict_proba", n_jobs=-1)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, oof)
    f2 = 5 * precision * recall / np.clip(4 * precision + recall, 1e-12, None)
    best = int(np.nanargmax(f2[:-1]))  # the last PR point has no threshold
    return float(thresholds[best]), {"oof_precision": float(precision[best]), "oof_recall": float(recall[best]), "oof_f2": float(f2[best])}
