"""1D Convolutional Neural Network over each calf's recent daily history.

The classical models (and the MLP) see one day at a time. BRD develops over
several days, so here every sample is a window of the calf's last WINDOW daily
records (the labelled day and the days before it), shape (features, WINDOW).
A 1D convolution slides along the time axis and can learn short-term trends
such as "milk intake dropping for three days".

Leakage safeguards
* windows only contain the labelled day and EARLIER days (never the future);
* the imputer/scaler are fitted on training calves only;
* train/validation/test splits are by calf, identical to the classical models.
"""
from __future__ import annotations

import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from torch import nn

WINDOW = 7
SEED = 42


class CNN1D(nn.Module):
    def __init__(self, n_features: int, channels: int = 32, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel_size=3, padding=1), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1), nn.Flatten(),
            nn.Linear(channels, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1),
        )

    def forward(self, x):  # x: (batch, features, time)
        return self.net(x).squeeze(-1)


def build_windows(raw: pd.DataFrame, modeled: pd.DataFrame, numeric: list[str], calf: str, fit_calves, window: int = WINDOW):
    """Return X (n, features+1, window) aligned with `modeled` rows.

    `raw` must be the cleaned full dataset (with every day of every calf) and
    `modeled` must carry the `_row` column pointing at the raw row index.
    The extra channel is a padding indicator (1 = no earlier record available).
    """
    order = raw.sort_values([calf, "Date"]).index
    fit_rows = raw.index[raw[calf].isin(fit_calves)]
    imputer = SimpleImputer(strategy="median").fit(raw.loc[fit_rows, numeric])
    scaler = StandardScaler().fit(imputer.transform(raw.loc[fit_rows, numeric]))
    values = pd.DataFrame(scaler.transform(imputer.transform(raw[numeric])), index=raw.index)
    position, history = {}, {}
    for calf_id, idx in raw.loc[order].groupby(calf, sort=False).groups.items():
        history[calf_id] = values.loc[idx].to_numpy(np.float32)
        for i, row in enumerate(idx):
            position[row] = (calf_id, i)
    X = np.zeros((len(modeled), len(numeric) + 1, window), dtype=np.float32)
    for n, row in enumerate(modeled["_row"].to_numpy()):
        calf_id, i = position[row]
        past = history[calf_id][max(0, i - window + 1): i + 1]
        X[n, :-1, window - len(past):] = past.T
        X[n, -1, : window - len(past)] = 1.0
    return X


def _train(X_tr, y_tr, X_va=None, y_va=None, epochs: int = 60, patience: int = 8, lr: float = 1e-3, weight_decay: float = 1e-4):
    """Weighted binary cross-entropy + Adam. Early stopping on validation loss if given."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = CNN1D(X_tr.shape[1])
    pos_weight = torch.tensor((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xt, yt = torch.tensor(X_tr), torch.tensor(y_tr, dtype=torch.float32)
    history, best_state, best_loss, best_epoch, wait = {"train": [], "valid": []}, None, np.inf, epochs, 0
    gen = torch.Generator().manual_seed(SEED)
    for epoch in range(epochs):
        model.train(); total = 0.0
        for batch in torch.randperm(len(Xt), generator=gen).split(128):
            opt.zero_grad(); loss = loss_fn(model(Xt[batch]), yt[batch]); loss.backward(); opt.step(); total += loss.item() * len(batch)
        history["train"].append(total / len(Xt))
        if X_va is not None:
            model.eval()
            with torch.no_grad():
                v = loss_fn(model(torch.tensor(X_va)), torch.tensor(y_va, dtype=torch.float32)).item()
            history["valid"].append(v)
            if v < best_loss - 1e-4:
                best_loss, best_state, best_epoch, wait = v, copy.deepcopy(model.state_dict()), epoch + 1, 0
            else:
                wait += 1
                if wait >= patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_epoch


def predict_proba(model, X) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X))).numpy()


def run_cnn(raw, train, test, numeric, calf, cv, figures_path):
    """Grouped-CV on training calves -> OOF probabilities, threshold, epochs; then refit and test."""
    torch.set_num_threads(4)
    X_train = build_windows(raw, train, numeric, calf, set(train[calf]))
    X_test = build_windows(raw, test, numeric, calf, set(train[calf]))
    y_train, groups = train["target"].to_numpy(), train[calf].to_numpy()

    oof, epochs, curves = np.zeros(len(train)), [], None
    for fold, (tr, va) in enumerate(cv.split(X_train, y_train, groups)):
        # re-fit the scaler on this fold's training calves only
        Xf = build_windows(raw, train, numeric, calf, set(groups[tr]))
        model, hist, best_epoch = _train(Xf[tr], y_train[tr], Xf[va], y_train[va])
        oof[va] = predict_proba(model, Xf[va]); epochs.append(best_epoch)
        curves = curves or hist
    ap = average_precision_score(y_train, oof)
    precision, recall, thresholds = precision_recall_curve(y_train, oof)
    f2 = 5 * precision * recall / np.clip(4 * precision + recall, 1e-12, None)
    best = int(np.nanargmax(f2[:-1])); threshold = float(thresholds[best])

    final_epochs = int(np.median(epochs))
    model, _, _ = _train(X_train, y_train, epochs=final_epochs)
    prob = predict_proba(model, X_test)

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(curves["train"]) + 1), curves["train"], label="Training loss", color="#3b6378")
    plt.plot(range(1, len(curves["valid"]) + 1), curves["valid"], label="Validation loss (unseen calves)", color="#c95c4b")
    plt.axvline(epochs[0], ls="--", c="grey", label=f"Early-stopping epoch ({epochs[0]})")
    plt.title("1D-CNN: weighted BCE loss per epoch (CV fold 1)"); plt.xlabel("Epoch"); plt.ylabel("Weighted BCE loss"); plt.legend(); plt.tight_layout()
    plt.savefig(figures_path / "cnn_loss_curve.png", dpi=180); plt.close()

    info = {"window_days": WINDOW, "input_shape": [int(X_train.shape[1]), WINDOW], "cv_average_precision": float(ap), "threshold": threshold,
            "fold_best_epochs": epochs, "final_epochs": final_epochs, "n_parameters": int(sum(p.numel() for p in model.parameters()))}
    print(f"1D-CNN: CV AP={ap:.3f} | threshold={threshold:.3f} | epochs per fold={epochs}")
    return model, prob, info, X_test


def cnn_grouped_cv(raw, modeled, numeric, calf, cv, threshold, epochs):
    """5-fold calf-grouped CV on all calves (same folds as the classical models)."""
    from .evaluate import score_predictions
    y, groups = modeled["target"].to_numpy(), modeled[calf].to_numpy()
    scores = []
    for tr, va in cv.split(modeled, y, groups):
        X = build_windows(raw, modeled, numeric, calf, set(groups[tr]))
        model, _, _ = _train(X[tr], y[tr], epochs=epochs)
        prob = predict_proba(model, X[va])
        scores.append(score_predictions(y[va], (prob >= threshold).astype(int), prob))
    return scores
