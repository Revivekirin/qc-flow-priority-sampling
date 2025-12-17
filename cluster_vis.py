import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

import hdbscan


def build_traj_features(
    dataset: Dict[str, np.ndarray],
    trajectory_boundaries: List[Tuple[int, int]],
    obs_key: str = "observations",
    obs_slice: Optional[slice] = None,
    k_keyframes: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(dataset[obs_key])
    if obs_slice is not None:
        obs = obs[..., obs_slice]

    num_traj = len(trajectory_boundaries)
    features = []
    lengths = []

    for (s, e) in trajectory_boundaries:
        T = int(e - s + 1)
        lengths.append(T)

        if T <= 1:
            idx = np.array([s] * k_keyframes, dtype=int)
        else:
            idx = np.linspace(s, e, k_keyframes, dtype=int)

        pts = obs[idx]
        pts_flat = pts.reshape(k_keyframes, -1)

        v = np.diff(pts_flat, axis=0)
        v_norm = np.linalg.norm(v, axis=-1)
        v_mean = float(v_norm.mean()) if v_norm.size > 0 else 0.0
        v_std = float(v_norm.std()) if v_norm.size > 0 else 0.0

        feat = np.concatenate([
            np.array([T, v_mean, v_std], dtype=np.float32),
            pts_flat.flatten().astype(np.float32),
        ])
        features.append(feat)

    features = np.stack(features, axis=0).astype(np.float32)
    lengths = np.asarray(lengths, dtype=np.int32)

    return features, lengths


def visualize_traj_clusters(
    X_pca,
    cluster_ids,
    lengths=None,
    save_prefix="traj_pca",
    noise_label=-1,
):
    cluster_ids = np.asarray(cluster_ids)
    unique_labels = np.unique(cluster_ids)

    has_noise = noise_label in unique_labels
    non_noise_mask = cluster_ids != noise_label
    noise_mask = cluster_ids == noise_label

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_pca[non_noise_mask, 0],
        X_pca[non_noise_mask, 1],
        c=cluster_ids[non_noise_mask],
        cmap="tab20",
        s=20,
    )
    cbar = plt.colorbar(scatter, label="Cluster ID")

    if has_noise:
        plt.scatter(
            X_pca[noise_mask, 0],
            X_pca[noise_mask, 1],
            c="lightgray",
            s=15,
            alpha=0.6,
            label="Noise",
        )
        plt.legend()

    plt.title("Trajectory PCA - clustering")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(f"plot/{save_prefix}_cluster.png", dpi=200)
    plt.close()

    if lengths is not None:
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=lengths,
            cmap="viridis",
            s=20,
        )
        plt.colorbar(scatter, label="Trajectory Length")
        plt.title("Trajectory PCA - colored by length")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.tight_layout()
        plt.savefig(f"plot/{save_prefix}_length.png", dpi=200)
        plt.close()

    valid_labels = unique_labels[unique_labels != noise_label]
    centroids = []

    for cid in valid_labels:
        pts = X_pca[cluster_ids == cid]
        if len(pts) == 0:
            continue
        centroids.append(pts.mean(axis=0))

    centroids = np.array(centroids)

    plt.figure(figsize=(8, 6))
    plt.scatter(
        X_pca[non_noise_mask, 0],
        X_pca[non_noise_mask, 1],
        c=cluster_ids[non_noise_mask],
        cmap="tab20",
        s=18,
        alpha=0.6,
    )
    if has_noise:
        plt.scatter(
            X_pca[noise_mask, 0],
            X_pca[noise_mask, 1],
            c="lightgray",
            s=15,
            alpha=0.5,
        )

    if len(centroids) > 0:
        plt.scatter(
            centroids[:, 0],
            centroids[:, 1],
            c="black",
            s=80,
            marker="x",
            label="Cluster Centroids",
        )
        plt.legend()

    plt.title("Trajectory PCA - cluster centroids")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(f"plot/{save_prefix}_centroids.png", dpi=200)
    plt.close()

    print(f"[Saved] {save_prefix}_cluster.png")
    if lengths is not None:
        print(f"[Saved] {save_prefix}_length.png")
    print(f"[Saved] {save_prefix}_centroids.png")


def plot_elbow(X, k_min=2, k_max=16, save_path="traj_pca_elbow.png"):
    Ks = list(range(k_min, k_max + 1))
    inertias = []

    for K in Ks:
        print(f"[Elbow] Fitting KMeans with K={K} ...")
        kmeans = KMeans(n_clusters=K, random_state=0, n_init="auto")
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(Ks, inertias, marker="o")
    plt.xlabel("Number of clusters K")
    plt.ylabel("Inertia (within-cluster SSE)")
    plt.title("Elbow plot for K selection")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"[Saved] Elbow figure → {save_path}")
    print("Ks:", Ks)
    print("Inertias:", inertias)
    return Ks, inertias


def select_k_elbow(Ks, inertias):
    inertias = np.array(inertias)
    Ks = np.array(Ks)

    d1 = np.diff(inertias)
    d2 = np.diff(d1)

    elbow_idx = np.argmin(d2)
    best_k = Ks[elbow_idx + 2]
    return int(best_k)


def select_k_by_value_homogeneity(
    X,
    traj_returns,
    k_candidates,
    min_cluster_size=30,
    lambda_small=1.0,
    lambda_k=0.02,
):
    best_k = None
    best_score = np.inf

    N = len(traj_returns)
    traj_returns = np.asarray(traj_returns, dtype=np.float64)

    for K in k_candidates:
        km = KMeans(n_clusters=K, random_state=0, n_init=20)
        labels = km.fit_predict(X)

        var_weighted_sum = 0.0
        total_weight = 0.0
        small_cnt = 0

        for c in range(K):
            idx = (labels == c)
            n_c = int(idx.sum())
            if n_c <= 1:
                small_cnt += 1
                continue

            r = traj_returns[idx]
            var_c = float(np.var(r))

            var_weighted_sum += n_c * var_c
            total_weight += n_c

            if n_c < min_cluster_size:
                small_cnt += 1

        if total_weight == 0:
            continue

        var_weighted_mean = var_weighted_sum / total_weight
        p_small = small_cnt / float(K)
        penalty_small = lambda_small * p_small
        penalty_k = lambda_k * K

        score = var_weighted_mean + penalty_small + penalty_k

        if score < best_score:
            best_score = score
            best_k = K

    return best_k


if __name__ == '__main__':
    train_dataset = ...
    trajectory_boundaries = ...
    env_name = ...

    features, lengths = build_traj_features(
        train_dataset,
        trajectory_boundaries,
        obs_key="observations",
        obs_slice=None,
        k_keyframes=20
    )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features).astype(np.float64)
    pca_2d = PCA(n_components=2, random_state=0)
    X_pca = pca_2d.fit_transform(X_scaled)

    Ks, inertias = plot_elbow(X_scaled, k_min=2, k_max=20, save_path=f"plot/{env_name}_elbow.png")
    K = select_k_elbow(Ks, inertias)

    kmeans = KMeans(n_clusters=K, random_state=0)
    cluster_ids = kmeans.fit_predict(X_scaled)
    visualize_traj_clusters(X_pca, cluster_ids, lengths, save_prefix=f"{env_name}_kmeans")
    print(f"kmeans pngs are saved!")

    clusterer = hdbscan.HDBSCAN(min_cluster_size=30)
    labels = clusterer.fit_predict(X_scaled)
    visualize_traj_clusters(X_pca, labels, lengths=lengths, save_prefix=f"{env_name}_hdbscan")
    print(f"hdbsan pngs are saved!")

    pca_gmm = PCA(n_components=min(20, X_scaled.shape[1]), random_state=0)
    X_gmm = pca_gmm.fit_transform(X_scaled)
    gmm = GaussianMixture(n_components=K, covariance_type="full", random_state=0)
    gmm_labels = gmm.fit_predict(X_scaled)
    visualize_traj_clusters(X_pca, gmm_labels, lengths=lengths, save_prefix=f"{env_name}_gmm", noise_label=-1)
    print("GMM pngs are saved!")

    spectral = SpectralClustering(
        n_clusters=K,
        assign_labels="kmeans",
        affinity="nearest_neighbors",
        n_neighbors=10,
        random_state=0,
    )
    spectral_labels = spectral.fit_predict(X_scaled)
    visualize_traj_clusters(X_pca, spectral_labels, lengths=lengths, save_prefix=f"{env_name}_gmm_spectral", noise_label=-1)
    print("Spectral pngs are saved!")