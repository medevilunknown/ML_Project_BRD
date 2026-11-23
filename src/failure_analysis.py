"""Failure-case analysis on the unseen test calves.

Answers: WHICH records/calves does the model get wrong, HOW confident was it,
and WHAT distinguishes the errors from the correct predictions?
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROFILE_FEATURES = ["Milk_Intake", "Milk_percent", "Speed", "Starter_Intake", "Total_Steps", "Lying_time", "Lying_bouts",
                    "Temp", "Temp_score", "Breath", "Fecal", "Weight", "Feederday", "Rewarded", "Unrewarded"]


def outcome_labels(y, pred):
    return np.select([(y == 1) & (pred == 1), (y == 1) & (pred == 0), (y == 0) & (pred == 1)], ["TP", "FN", "FP"], "TN")


def run_failure_analysis(train, test, predictions, thresholds, best_name, calf, figures_path, results_path) -> dict:
    y = test["target"].to_numpy()
    best = predictions[best_name]
    prob, pred = best["prob"], best["pred"]
    t = thresholds[best_name]
    frame = test.copy()
    frame["prob"], frame["pred"], frame["outcome"] = prob, pred, outcome_labels(y, pred)
    counts = frame["outcome"].value_counts().reindex(["TP", "FN", "FP", "TN"], fill_value=0)

    # 1. Probability distributions: how separable are the classes?
    plt.figure(figsize=(8, 4.5))
    bins = np.linspace(0, 1, 41)
    plt.hist(prob[y == 0], bins=bins, alpha=.6, density=True, label="Healthy/control", color="#3b6378")
    plt.hist(prob[y == 1], bins=bins, alpha=.6, density=True, label="Pre-BRD", color="#c95c4b")
    plt.axvline(t, c="k", ls="--", label=f"Decision threshold {t:.2f}")
    plt.title(f"{best_name}: predicted probability by true class (test calves)"); plt.xlabel("Predicted pre-BRD probability"); plt.ylabel("Density"); plt.legend()
    plt.tight_layout(); plt.savefig(figures_path / "failure_probability_overlap.png", dpi=180); plt.close()

    # 2. What do errors look like? Feature means as z-scores against healthy TRAINING records.
    feats = [f for f in PROFILE_FEATURES if f in frame.columns]
    ref = train.loc[train.target == 0, feats]
    z = (frame.groupby("outcome")[feats].mean() - ref.mean()) / ref.std().replace(0, np.nan)
    z = z.reindex(["TP", "FN", "FP", "TN"]).dropna(how="all")
    plt.figure(figsize=(12, 3.8))
    sns.heatmap(z, cmap="vlag", center=0, annot=True, fmt=".1f", cbar_kws={"label": "z-score vs healthy training records"})
    plt.title(f"Feature profile of correct vs wrong predictions ({best_name})"); plt.ylabel("Outcome"); plt.tight_layout()
    plt.savefig(figures_path / "failure_feature_profiles.png", dpi=180); plt.close()
    z.to_csv(results_path / "failure_feature_profiles.csv")

    # 3. Age confound: error rates by days on the feeder.
    frame["age_band"] = pd.cut(frame["Feederday"], bins=[0, 14, 28, 42, 56, 90], labels=["1-14", "15-28", "29-42", "43-56", "57+"])
    by_age = frame.groupby("age_band", observed=False).apply(lambda g: pd.Series({
        "healthy_records": int((g.target == 0).sum()), "false_alarm_rate": g.loc[g.target == 0, "pred"].mean(),
        "pre_brd_records": int((g.target == 1).sum()), "recall": g.loc[g.target == 1, "pred"].mean()}), include_groups=False)
    by_age.to_csv(results_path / "failure_by_age.csv")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs = np.arange(len(by_age))
    ax.bar(xs - .2, by_age["false_alarm_rate"].fillna(0), .4, label="False-alarm rate (healthy)", color="#3b6378")
    ax.bar(xs + .2, by_age["recall"].fillna(0), .4, label="Recall (pre-BRD)", color="#c95c4b")
    for x, n in zip(xs, by_age["pre_brd_records"]):
        ax.text(x + .2, 0.02, f"n={int(n)}", ha="center", fontsize=8, color="white")
    ax.set_xticks(xs, by_age.index); ax.set(xlabel="Days on automatic feeder (age proxy)", ylabel="Rate", ylim=(0, 1), title="Errors by calf age band (test calves)")
    ax.legend(); plt.tight_layout(); plt.savefig(figures_path / "failure_by_age.png", dpi=180); plt.close()

    # 3b. Age-matched discrimination: compare sick vs healthy records of the SAME age range
    #     (all pre-BRD records occur in the first 42 days on the feeder).
    from sklearn.metrics import roc_auc_score
    young = (frame["Feederday"] <= 42).to_numpy()
    age_matched_auc = {n: float(roc_auc_score(y[young], predictions[n]["prob"][young])) for n in predictions}
    pd.Series(age_matched_auc, name="ROC-AUC (Feederday<=42)").to_csv(results_path / "age_matched_auc.csv")

    # 4. Calf-level view: is every sick calf caught at least once in its 6 pre-diagnosis days?
    calves = frame.groupby(calf).agg(status=("target", "max"), records=("target", "size"), flagged=("pred", "sum"),
                                     max_prob=("prob", "max"), mean_prob=("prob", "mean"), mean_feederday=("Feederday", "mean"))
    calves["status"] = calves["status"].map({1: "Pre-BRD", 0: "Healthy"})
    calves["flag_rate"] = calves["flagged"] / calves["records"]
    calves = calves.sort_values(["status", "max_prob"])
    calves.to_csv(results_path / "failure_by_calf.csv")
    sick, healthy = calves[calves.status == "Pre-BRD"], calves[calves.status == "Healthy"]

    # 5. Model agreement on the pre-BRD test records: hard cases are missed by every model.
    names = list(predictions)
    hits = pd.DataFrame({n: predictions[n]["pred"] for n in names}, index=frame.index)[y == 1]
    hits[calf] = frame.loc[y == 1, calf]
    per_calf = hits.groupby(calf)[names].mean()
    plt.figure(figsize=(9, max(3, .35 * len(per_calf) + 1)))
    sns.heatmap(per_calf, cmap="Reds", vmin=0, vmax=1, annot=True, fmt=".2f", cbar_kws={"label": "share of the 6 pre-diagnosis days flagged"})
    plt.title("Recall per sick test calf and model"); plt.ylabel("Calf (SENDER)"); plt.xlabel(""); plt.xticks(rotation=25, ha="right"); plt.tight_layout()
    plt.savefig(figures_path / "failure_model_agreement.png", dpi=180); plt.close()
    n_models_hit = hits[names].sum(axis=1)

    # 6. Missing data: are errors more common when sensor values are missing?
    feature_cols = [c for c in feats if c in frame]
    frame["has_missing"] = frame[feature_cols].isna().any(axis=1)
    missing = frame.groupby("has_missing").apply(lambda g: (g.pred != g.target).mean(), include_groups=False)

    summary = {
        "model": best_name, "threshold": t, "confusion": counts.to_dict(),
        "mean_prob": {"pre_brd": float(prob[y == 1].mean()), "healthy": float(prob[y == 0].mean())},
        "fn_high_confidence_healthy": int(((frame.outcome == "FN") & (frame.prob < t / 2)).sum()),
        "fp_high_confidence": int(((frame.outcome == "FP") & (frame.prob > (1 + t) / 2)).sum()),
        "sick_calves": int(len(sick)), "sick_calves_never_flagged": int((sick.flagged == 0).sum()),
        "sick_calves_flagged_3plus_days": int((sick.flagged >= 3).sum()),
        "never_flagged_calves": sick.index[sick.flagged == 0].astype("int64").tolist(),
        "healthy_calves": int(len(healthy)), "healthy_calves_with_false_alarm": int((healthy.flagged > 0).sum()),
        "worst_healthy_calf_flag_rate": float(healthy.flag_rate.max()),
        "by_age": by_age.round(3).to_dict(orient="index"),
        "pre_brd_records_missed_by_all_models": int((n_models_hit == 0).sum()),
        "pre_brd_records_caught_by_all_models": int((n_models_hit == len(names)).sum()),
        "pre_brd_records": int(len(hits)),
        "error_rate_by_missing": {str(k): float(v) for k, v in missing.items()},
        "z_profiles": z.round(2).to_dict(orient="index"),
        "age_matched_auc": age_matched_auc,
    }
    print(f"\nFAILURE ANALYSIS ({best_name}): {counts.to_dict()} | sick calves never flagged: {summary['sick_calves_never_flagged']}/{len(sick)}"
          f" | healthy calves with a false alarm: {summary['healthy_calves_with_false_alarm']}/{len(healthy)}")
    return summary
