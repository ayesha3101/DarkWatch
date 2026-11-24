"""Do boats fall into natural movement modes? Unsupervised learning (k-means).

Run:  python src/clustering.py

Uses the same 15-minute window features as the classifier, but NO labels: the numbers are
standardised (each feature scaled to average 0 and spread 1, so speed in knots does not
swamp the others) and k-means groups similar windows. We try k = 2 to 6 and keep the k with
the best silhouette score (how well each window fits its own cluster compared with the next
nearest one: 1 is perfect, 0 means overlapping, negative means likely mis-grouped).
Afterwards we look at the real labels to INTERPRET the clusters (adjusted Rand index:
1 = clusters match the labels exactly, 0 = no better than chance).

k-means parameters: n_clusters = k, n_init = 10 random starts (the best of 10 is kept),
random_state = 42 so the result can be repeated.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from classifier import FEATURES, make_windows
from movement import TRACKS_FILE

FIG_DIR, RESULT_DIR = Path("outputs/figures"), Path("outputs/results")
SEED, KS = 42, range(2, 7)
SHADES = ["#2e8b57", "#111111", "#9bd0b0", "#777777", "#0d4d2a", "#cccccc"]


def scale(windows):
    """Standardise the movement features."""
    return StandardScaler().fit_transform(windows[FEATURES])


def choose_k(x, ks=KS):
    """Try k-means for each k. Returns (table with silhouette scores, best k)."""
    rows = []
    for k in ks:
        model = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(x)
        rows.append({"k": k, "silhouette": silhouette_score(x, model.labels_), "inertia": model.inertia_})
    table = pd.DataFrame(rows)
    return table.round(3), int(table.loc[table["silhouette"].idxmax(), "k"])


def profile(windows, clusters):
    """What each cluster looks like: size, typical movement, and (for interpretation only) the share of fishing windows."""
    frame = windows.assign(cluster=clusters)
    out = frame.groupby("cluster").agg(windows=("label", "size"), boats=("vessel_id", "nunique"),
                                       mean_speed=("mean_speed", "mean"), share_stopped=("share_stopped", "mean"),
                                       mean_abs_turn=("mean_abs_turn", "mean"),
                                       share_fishing=("label", lambda s: (s == "fishing").mean()))
    return out.round(3)


def make_charts(x, clusters, k_table):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    points = PCA(n_components=2, random_state=SEED).fit_transform(x)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for c in sorted(set(clusters)):
        ax.scatter(*points[clusters == c].T, s=18, color=SHADES[c % len(SHADES)], edgecolor="black", linewidth=0.3, label=f"cluster {c}")
    ax.set_xlabel("first main direction of variation")
    ax.set_ylabel("second main direction of variation")
    ax.set_title("Movement windows grouped by k-means (no labels used)")
    ax.legend(fontsize=8)
    fig.savefig(FIG_DIR / "clustering_pca.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(k_table["k"], k_table["silhouette"], marker="o", color="#2e8b57")
    ax.set_xlabel("number of clusters k")
    ax.set_ylabel("silhouette score")
    ax.set_title("Choosing k")
    fig.savefig(FIG_DIR / "clustering_silhouette.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    try:
        tracks = pd.read_csv(TRACKS_FILE, parse_dates=["time"])
    except FileNotFoundError:
        sys.exit(f"{TRACKS_FILE} not found. Run python src/clean_tracks.py first.")
    windows = make_windows(tracks)
    if len(windows) < 20:
        sys.exit("Not enough windows to cluster.")
    x = scale(windows)
    k_table, best_k = choose_k(x)
    print(f"{len(windows)} windows from {windows['vessel_id'].nunique()} boats. Features: {', '.join(FEATURES)}\n")
    print(k_table.to_string(index=False))
    clusters = KMeans(n_clusters=best_k, n_init=10, random_state=SEED).fit_predict(x)
    print(f"\nBest k by silhouette: {best_k}. Clusters, described by how they move:\n")
    prof = profile(windows, clusters)
    print(prof.to_string())
    print(f"\nAdjusted Rand index against the real fishing / other labels: {adjusted_rand_score(windows['label'], clusters):.3f}")
    print("\nWindows per cluster and real label:")
    print(pd.crosstab(clusters, windows["label"], rownames=["cluster"]).to_string())

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    k_table.to_csv(RESULT_DIR / "clustering_k.csv", index=False)
    prof.to_csv(RESULT_DIR / "clustering_profiles.csv")
    make_charts(x, clusters, k_table)
    print(f"\nSaved 2 tables in {RESULT_DIR}/ and 2 charts in {FIG_DIR}/")


if __name__ == "__main__":
    main()