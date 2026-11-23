"""Unsupervised analysis: PCA + K-Means clustering of daily calf records.

Question: without using the labels, do the behavioural/feeding records form
natural groups, and is the pre-BRD rate different between those groups?
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run_clustering(modeled: pd.DataFrame, numeric: list[str], figures_path, results_path, k_range=range(2, 9), variance_to_keep: float = 0.90, random_state: int = 42) -> dict:
    X = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]).fit_transform(modeled[numeric])

    # PCA: keep enough components to explain `variance_to_keep` of the variance.
    pca_full = PCA(random_state=random_state).fit(X)
    cumulative = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumulative, variance_to_keep) + 1)
    Z = PCA(n_components=n_components, random_state=random_state).fit_transform(X)

    # K-Means for several k; choose k with the highest silhouette score.
    rng = np.random.default_rng(random_state)
    sample = rng.choice(len(Z), size=min(3000, len(Z)), replace=False)  # silhouette is O(n^2)
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(Z)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": silhouette_score(Z[sample], km.labels_[sample])})
    scores = pd.DataFrame(rows)
    best_k = int(scores.loc[scores.silhouette.idxmax(), "k"])
    labels = KMeans(n_clusters=best_k, n_init=10, random_state=random_state).fit_predict(Z)

    # Relate clusters back to the (unused) label and to the original features.
    frame = modeled[numeric].copy()
    frame["cluster"] = labels
    frame["target"] = modeled["target"].values
    profile = frame.groupby("cluster").agg(records=("target", "size"), pre_brd_rate=("target", "mean"))
    key = [c for c in ["Milk_Intake", "Speed", "Lying_time", "Total_Steps", "Starter_Intake", "Temp", "Weight", "Feederage"] if c in numeric]
    profile = profile.join(frame.groupby("cluster")[key].mean())
    profile.to_csv(results_path / "cluster_profiles.csv")
    scores.to_csv(results_path / "clustering_scores.csv", index=False)

    # Figures
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(range(1, len(cumulative) + 1), cumulative, marker="o", ms=3)
    axes[0].axhline(variance_to_keep, ls="--", c="grey"); axes[0].axvline(n_components, ls="--", c="grey")
    axes[0].set(title=f"PCA: {n_components} components keep {cumulative[n_components-1]:.0%} variance", xlabel="Components", ylabel="Cumulative explained variance")
    axes[1].plot(scores.k, scores.inertia, marker="o"); axes[1].set(title="Elbow method", xlabel="k", ylabel="Inertia (within-cluster SS)")
    axes[2].plot(scores.k, scores.silhouette, marker="o", c="#c95c4b"); axes[2].set(title=f"Silhouette score (best k={best_k})", xlabel="k", ylabel="Silhouette")
    plt.tight_layout(); plt.savefig(figures_path / "clustering_selection.png", dpi=180); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    sns.scatterplot(x=Z[:, 0], y=Z[:, 1], hue=labels, palette="tab10", s=8, alpha=.6, ax=axes[0], legend="full")
    axes[0].set(title=f"K-Means clusters (k={best_k}) on first two PCs", xlabel="PC1", ylabel="PC2")
    order = np.argsort(modeled["target"].values)  # draw positives on top
    sns.scatterplot(x=Z[order, 0], y=Z[order, 1], hue=modeled["target"].values[order], palette={0: "#9bb7c9", 1: "#c95c4b"}, s=8, alpha=.7, ax=axes[1])
    axes[1].set(title="Same projection coloured by true label (1 = pre-BRD)", xlabel="PC1", ylabel="PC2")
    plt.tight_layout(); plt.savefig(figures_path / "clusters_pca.png", dpi=180); plt.close()

    summary = {"pca_components": n_components, "pca_variance_kept": float(cumulative[n_components - 1]), "best_k": best_k,
               "best_silhouette": float(scores.silhouette.max()), "cluster_pre_brd_rate": profile["pre_brd_rate"].round(4).to_dict(),
               "cluster_sizes": profile["records"].to_dict()}
    print(f"\nCLUSTERING: PCA kept {n_components} components ({cumulative[n_components-1]:.1%} variance); best k={best_k} (silhouette={scores.silhouette.max():.3f})")
    print(profile[["records", "pre_brd_rate"]].to_string())
    return summary
