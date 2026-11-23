"""Train, evaluate, visualize, and save the BRD models."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
from sklearn.pipeline import Pipeline

from .data_preprocessing import create_target, load_dataset, make_preprocessor, select_features, split_by_calf
from .clustering import run_clustering
from .evaluate import evaluate_models, grouped_cross_validate, plot_boosting_loss, plot_learning_curve
from .deep_models import cnn_grouped_cv, run_cnn
from .failure_analysis import run_failure_analysis
from .tuning import group_cv, model_grids, tune_model, tune_threshold
from .data_preprocessing import clean_columns
from .utils import DATA_PATH, FIGURES_PATH, MODELS_PATH, RESULTS_PATH, ensure_output_dirs

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")


def save_dataset_figures(raw, modeled, feature_columns):
    sns.set_theme(style="whitegrid", context="notebook")
    plt.figure(figsize=(6, 4))
    modeled["target"].map({0: "Healthy/control", 1: "Pre-BRD"}).value_counts().plot.bar(color=["#3b7a57", "#c95c4b"])
    plt.title("Class distribution"); plt.ylabel("Daily records"); plt.tight_layout(); plt.savefig(FIGURES_PATH / "class_distribution.png", dpi=180); plt.close()
    missing = raw.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    plt.figure(figsize=(10, 5)); missing.plot.bar(color="#3b6378"); plt.title("Missing values by column"); plt.ylabel("Missing cells"); plt.xticks(rotation=70, ha="right"); plt.tight_layout(); plt.savefig(FIGURES_PATH / "missing_values.png", dpi=180); plt.close()
    numeric = [c for c in feature_columns if pd.api.types.is_numeric_dtype(modeled[c])][:8]
    if numeric:
        long = modeled[["target", *numeric]].copy().melt("target", var_name="Feature", value_name="Value")
        long["Class"] = long["target"].map({0: "Healthy/control", 1: "Pre-BRD"})
        plt.figure(figsize=(12, 6)); sns.boxplot(data=long, x="Feature", y="Value", hue="Class", showfliers=False); plt.xticks(rotation=45, ha="right"); plt.title("Behaviour and feeding features by class"); plt.tight_layout(); plt.savefig(FIGURES_PATH / "feature_comparisons.png", dpi=180); plt.close()
        plt.figure(figsize=(10, 8)); sns.heatmap(modeled[numeric].corr(), cmap="vlag", center=0, annot=False); plt.title("Correlation heatmap of selected numeric features"); plt.tight_layout(); plt.savefig(FIGURES_PATH / "correlation_heatmap.png", dpi=180); plt.close()


def transformed_feature_names(fitted_pipeline):
    prep = fitted_pipeline.named_steps["preprocess"]
    return list(prep.get_feature_names_out())


def save_model_figures(models, results, predictions, X_test, y_test):
    plt.figure(figsize=(10, 5)); results.set_index("Model")[['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']].plot.bar(ax=plt.gca()); plt.ylim(0, 1.05); plt.title("Model comparison on unseen test calves (tuned thresholds)"); plt.ylabel("Score"); plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(FIGURES_PATH / "model_comparison.png", dpi=180); plt.close()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (name, values) in zip(axes.ravel(), predictions.items()):
        ConfusionMatrixDisplay.from_predictions(y_test, values["pred"], display_labels=["Healthy", "Pre-BRD"], cmap="Blues", ax=ax, colorbar=False); ax.set_title(name)
    plt.tight_layout(); plt.savefig(FIGURES_PATH / "confusion_matrices.png", dpi=180); plt.close()
    plt.figure(figsize=(7, 6))
    for name, values in predictions.items():
        if values["prob"] is not None and len(np.unique(y_test)) > 1:
            fpr, tpr, _ = roc_curve(y_test, values["prob"]); auc = results.loc[results.Model == name, "ROC-AUC"].iloc[0]; plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=.5); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curves"); plt.legend(); plt.tight_layout(); plt.savefig(FIGURES_PATH / "roc_curves.png", dpi=180); plt.close()
    for name, filename, component in [("Random Forest", "random_forest_importance.png", "rf"), ("Logistic Regression", "logistic_coefficients.png", "logistic")]:
        if name not in models: continue
        pipe = models[name]; names = transformed_feature_names(pipe)
        estimator = pipe.named_steps["model"]
        values = estimator.feature_importances_ if component == "rf" else estimator.coef_[0]
        top = pd.Series(values, index=names).sort_values(key=abs).tail(10)
        plt.figure(figsize=(8, 5)); top.sort_values().plot.barh(color="#c95c4b" if component == "logistic" else "#3b6378"); plt.title(f"Top features: {name}"); plt.tight_layout(); plt.savefig(FIGURES_PATH / filename, dpi=180); plt.close()


def run_project(data_path: str | Path = DATA_PATH):
    ensure_output_dirs()
    sns.set_theme(style="whitegrid", context="notebook")

    # 1. Load data, build the target, choose leakage-free features
    raw, detected = load_dataset(data_path)
    calf = detected["calf_id"]
    modeled, info = create_target(raw, calf, detected["pneuday"])
    feature_columns, numeric, categorical, excluded = select_features(modeled, calf, detected["pneuday"])
    save_dataset_figures(raw, modeled, feature_columns)

    # 2. Unsupervised analysis (PCA + K-Means) - labels are not used for fitting
    clustering = run_clustering(modeled, numeric, FIGURES_PATH, RESULTS_PATH)

    # 3. 80/20 split by calf: test calves are never seen during tuning or training
    train, test = split_by_calf(modeled, calf)
    X_train, y_train, g_train = train[feature_columns], train["target"], train[calf]
    X_test, y_test = test[feature_columns], test["target"]

    def pipeline(model):
        return Pipeline([("preprocess", make_preprocessor(numeric, categorical)), ("model", model)])

    # 4. Hyperparameter search + threshold tuning with calf-grouped 5-fold CV on training calves
    models, thresholds, tuning_rows = {}, {}, []
    for name, (estimator, grid) in model_grids().items():
        search = tune_model(pipeline(estimator), grid, X_train, y_train, g_train)
        models[name] = search.best_estimator_
        thresholds[name], oof = tune_threshold(search.best_estimator_, X_train, y_train, g_train)
        best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
        tuning_rows.append({"Model": name, "CV average precision": search.best_score_, "Best params": json.dumps(best_params, default=str),
                            "Tuned threshold": thresholds[name], **{f"OOF {k[4:]}": v for k, v in oof.items()}})
        print(f"{name}: best {best_params} | CV AP={search.best_score_:.3f} | threshold={thresholds[name]:.3f}")
    # 4b. 1D-CNN on 7-day windows of each calf's history (same calf split and threshold rule)
    raw_clean = clean_columns(raw)
    cnn_model, cnn_prob, cnn_info, _ = run_cnn(raw_clean, train, test, numeric, calf, group_cv(), FIGURES_PATH)
    thresholds["1D-CNN"] = cnn_info["threshold"]
    tuning_rows.append({"Model": "1D-CNN", "CV average precision": cnn_info["cv_average_precision"],
                        "Best params": json.dumps({"window_days": cnn_info["window_days"], "epochs": cnn_info["final_epochs"], "conv_channels": 32}),
                        "Tuned threshold": cnn_info["threshold"]})
    tuning = pd.DataFrame(tuning_rows).sort_values("CV average precision", ascending=False)
    tuning.to_csv(RESULTS_PATH / "hyperparameter_tuning.csv", index=False)

    # 5. Final evaluation on unseen test calves
    results, predictions = evaluate_models(models, X_test, y_test, thresholds, extra_probabilities={"1D-CNN": cnn_prob})
    results.to_csv(RESULTS_PATH / "model_results.csv", index=False)
    cv = grouped_cross_validate(models, modeled[feature_columns], modeled["target"], modeled[calf], thresholds)
    cnn_cv = cnn_grouped_cv(raw_clean, modeled, numeric, calf, group_cv(), cnn_info["threshold"], cnn_info["final_epochs"])
    cnn_row = {"Model": "1D-CNN"}
    for metric in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]:
        values = [f[metric] for f in cnn_cv]
        cnn_row[f"{metric} mean"], cnn_row[f"{metric} std"] = float(np.mean(values)), float(np.std(values))
    cv = pd.concat([cv, pd.DataFrame([cnn_row])], ignore_index=True)
    cv.to_csv(RESULTS_PATH / "group_cross_validation.csv", index=False)
    save_model_figures(models, results, predictions, X_test, y_test)

    # 6. Model selected on TRAINING-CV score only (choosing on the test set would be leakage)
    best_name = tuning.iloc[0]["Model"]
    best = results.set_index("Model").loc[best_name]
    curve = plot_learning_curve(models[best_name], X_train, y_train, g_train, FIGURES_PATH / "learning_curve.png", f"Learning curve: {best_name}")
    loss = plot_boosting_loss(models["Gradient Boosting"], X_train, y_train, g_train, FIGURES_PATH / "boosting_loss_curve.png")
    mlp_curve = plot_mlp_curve(models["MLP"])
    error_analysis = analyse_errors(test, predictions[best_name], calf)
    failure = run_failure_analysis(train, test, predictions, thresholds, best_name, calf, FIGURES_PATH, RESULTS_PATH)

    if best_name == "1D-CNN":  # the CNN needs history windows; keep the best tabular model for the demo app
        demo_name = tuning[tuning.Model != "1D-CNN"].iloc[0]["Model"]
    else:
        demo_name = best_name
    torch.save(cnn_model.state_dict(), MODELS_PATH / "cnn_1d.pt")
    artifact = {"pipeline": models[demo_name], "threshold": thresholds[demo_name], "feature_columns": feature_columns, "numeric_features": numeric,
                "categorical_features": categorical, "calf_id": calf, "pneuday": detected["pneuday"], "model_name": demo_name}
    joblib.dump(artifact, MODELS_PATH / "best_brd_model.joblib")
    summary = {"detected": detected, "dataset": {"rows": len(raw), "columns": len(raw.columns), "unique_calves": int(raw[calf].nunique())},
               "target": {k: v for k, v in info.items() if not k.endswith("_calves") or isinstance(v, int)},
               "features": {"used": feature_columns, "excluded": excluded},
               "split": {"train_calves": int(train[calf].nunique()), "test_calves": int(test[calf].nunique()), "train_rows": len(train), "test_rows": len(test),
                         "test_positive_rows": int(y_test.sum())},
               "best_model": best_name, "best_threshold": thresholds[best_name], "tuning": tuning.to_dict(orient="records"),
               "results": results.to_dict(orient="records"), "learning_curve": curve, "boosting_loss": loss,
               "error_analysis": error_analysis, "failure_analysis": failure, "clustering": clustering,
               "cnn": cnn_info, "mlp_curve": mlp_curve}
    (RESULTS_PATH / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n==============================\nBRD MACHINE LEARNING PROJECT\n==============================")
    print(f"Rows: {len(raw)} | Calves: {raw[calf].nunique()} | Positive calves: {info['positive_calves']} | Healthy calves: {info['healthy_calves_used']}")
    print(f"Train calves: {train[calf].nunique()} | Test calves: {test[calf].nunique()}")
    print("\nTEST-SET RESULTS (tuned threshold)\n----------------------------------------")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("----------------------------------------")
    print(f"Selected model (best training-CV average precision): {best_name}, threshold {thresholds[best_name]:.3f}")
    print(f"Test Recall: {best['Recall']:.3f} | F1: {best['F1']:.3f} | ROC-AUC: {best['ROC-AUC']:.3f}")
    print(f"\nModel saved to: {MODELS_PATH / 'best_brd_model.joblib'}")
    return summary


def plot_mlp_curve(pipeline):
    """Training loss and validation accuracy of the tuned MLP per epoch."""
    mlp = pipeline.named_steps["model"]
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(mlp.loss_curve_, color="#3b6378", label="Training log-loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Training log-loss")
    ax2 = ax1.twinx(); ax2.plot(mlp.validation_scores_, color="#c95c4b", label="Validation accuracy"); ax2.set_ylabel("Validation accuracy")
    ax1.axvline(len(mlp.loss_curve_) - 1 - mlp.n_iter_no_change, ls="--", c="grey")
    fig.legend(loc="upper center", bbox_to_anchor=(.5, .9)); ax1.set_title(f"MLP {mlp.hidden_layer_sizes}: loss and validation accuracy"); fig.tight_layout()
    fig.savefig(FIGURES_PATH / "mlp_loss_curve.png", dpi=180); plt.close(fig)
    return {"epochs": int(mlp.n_iter_), "final_loss": float(mlp.loss_curve_[-1]), "best_validation_accuracy": float(mlp.best_validation_score_)}


def analyse_errors(test, prediction, calf):
    """Break test errors down by days-before-diagnosis and by calf."""
    frame = test[[calf, "Pneuday", "target"]].copy()
    frame["pred"] = prediction["pred"]
    pos = frame[frame.target == 1]
    by_day = pos.groupby("Pneuday")["pred"].agg(["size", "mean"]).rename(columns={"size": "records", "mean": "recall"})
    calves_pos = pos.groupby(calf)["pred"].max()
    calves_neg = frame[frame.target == 0].groupby(calf)["pred"].mean()
    plt.figure(figsize=(7, 4.5))
    by_day["recall"].plot.bar(color="#c95c4b"); plt.title("Recall by days before diagnosis (test calves)"); plt.xlabel("Pneuday (days relative to diagnosis)"); plt.ylabel("Recall"); plt.ylim(0, 1)
    plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(FIGURES_PATH / "recall_by_day.png", dpi=180); plt.close()
    by_day.to_csv(RESULTS_PATH / "recall_by_day.csv")
    return {"recall_by_pneuday": by_day["recall"].round(3).to_dict(), "records_by_pneuday": by_day["records"].to_dict(),
            "sick_test_calves": int(len(calves_pos)), "sick_calves_flagged_at_least_once": int(calves_pos.sum()),
            "healthy_test_calves": int(len(calves_neg)), "mean_false_alarm_rate_per_healthy_calf": float(calves_neg.mean()),
            "healthy_calves_with_any_false_alarm": int((calves_neg > 0).sum())}
