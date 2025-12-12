import numpy as np
from typing import List, Tuple, Optional, Dict

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering   # Spectral 추가
from sklearn.mixture import GaussianMixture              # GMM 추가

def build_traj_features(
    dataset: Dict[str, np.ndarray],
    trajectory_boundaries: List[Tuple[int, int]],
    obs_key: str = "observations",
    obs_slice: Optional[slice] = None,
    k_keyframes: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Trajectory-level feature를 만드는 함수.
    길이 + downsample된 경로 + 속도통계로 feature 벡터를 구성한다.

    Args:
        dataset:
            {"observations": np.ndarray, ...} 형태의 dict 또는 Dataset-like 객체.
            dataset[obs_key] 의 shape은 (N, obs_dim...) 이라고 가정.
        trajectory_boundaries:
            [(start0, end0), (start1, end1), ...] 형태의 리스트.
        obs_key:
            경로를 추출할 관측 key 이름 (기본: "observations").
            예: "ee_pos", "qpos", "observations" 등.
        obs_slice:
            관측 벡터에서 일부 차원만 쓰고 싶을 때 사용.
            예: obs_slice = slice(0, 3)  → obs[..., :3] 만 사용 (EE pos).
            None이면 전체 obs 사용.
        k_keyframes:
            각 traj에서 균일하게 뽑을 keyframe 개수.

    Returns:
        features: shape (num_traj, D) 의 feature matrix.
        lengths:  shape (num_traj,) 의 trajectory 길이 배열.
    """
    obs = np.asarray(dataset[obs_key])
    if obs_slice is not None:
        obs = obs[..., obs_slice]   # 예: (N, 3)

    num_traj = len(trajectory_boundaries)
    features = []
    lengths = []

    for (s, e) in trajectory_boundaries:
        # 1) 길이
        T = int(e - s + 1)
        lengths.append(T)

        # trajectory 길이가 너무 짧으면 (예: single step) 스킵 or 최소 보정
        if T <= 1:
            # padding 비슷하게: 같은 obs를 반복
            idx = np.array([s] * k_keyframes, dtype=int)
        else:
            # 2) 균일하게 k_keyframes 개 index 뽑기
            idx = np.linspace(s, e, k_keyframes, dtype=int)

        pts = obs[idx]                   # (K, obs_dim...)
        pts_flat = pts.reshape(k_keyframes, -1)  # (K, D_obs)

        # 3) 속도 norm 통계 (각 step의 L2 norm)
        v = np.diff(pts_flat, axis=0)          # (K-1, D_obs)
        v_norm = np.linalg.norm(v, axis=-1)    # (K-1,)
        v_mean = float(v_norm.mean()) if v_norm.size > 0 else 0.0
        v_std  = float(v_norm.std())  if v_norm.size > 0 else 0.0

        # 4) feature vector: [T, v_mean, v_std, flattened keyframes]
        feat = np.concatenate([
            np.array([T, v_mean, v_std], dtype=np.float32),
            pts_flat.flatten().astype(np.float32),
        ])
        features.append(feat)

    features = np.stack(features, axis=0).astype(np.float32)
    lengths = np.asarray(lengths, dtype=np.int32)

    return features, lengths

import matplotlib.pyplot as plt
import numpy as np

# X_pca: shape (num_traj, 2)
# cluster_ids: shape (num_traj,)

import numpy as np
import matplotlib.pyplot as plt

def visualize_traj_clusters(
    X_pca,
    cluster_ids,
    lengths=None,
    save_prefix="traj_pca",
    noise_label=-1,
):
    """
    X_pca      : (N, 2) PCA 결과
    cluster_ids: (N,) 클러스터 라벨 (KMeans의 labels_ 또는 HDBSCAN의 labels_)
    lengths    : (N,) trajectory 길이 (옵션)
    noise_label: HDBSCAN noise 라벨 (기본 -1, KMeans면 무시됨)
    """

    cluster_ids = np.asarray(cluster_ids)
    unique_labels = np.unique(cluster_ids)

    # HDBSCAN의 noise(-1)를 따로 분리
    has_noise = noise_label in unique_labels
    non_noise_mask = cluster_ids != noise_label
    noise_mask = cluster_ids == noise_label

    # -----------------------------
    # 1) 클러스터 결과 시각화
    # -----------------------------
    plt.figure(figsize=(8, 6))

    # non-noise 포인트
    scatter = plt.scatter(
        X_pca[non_noise_mask, 0],
        X_pca[non_noise_mask, 1],
        c=cluster_ids[non_noise_mask],
        cmap="tab20",
        s=20,
    )
    cbar = plt.colorbar(scatter, label="Cluster ID")

    # noise 포인트는 회색으로 별도 표시
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
    plt.savefig(f"{save_prefix}_cluster.png", dpi=200)
    plt.close()

    # -----------------------------
    # 2) 길이 기반 시각화
    # -----------------------------
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
        plt.savefig(f"{save_prefix}_length.png", dpi=200)
        plt.close()

    # -----------------------------
    # 3) cluster별 centroid 시각화
    #    (noise는 centroid에서 제외)
    # -----------------------------
    valid_labels = unique_labels[unique_labels != noise_label]
    centroids = []

    for cid in valid_labels:
        pts = X_pca[cluster_ids == cid]
        if len(pts) == 0:
            continue
        centroids.append(pts.mean(axis=0))

    centroids = np.array(centroids)

    plt.figure(figsize=(8, 6))
    # 전체 포인트 (noise 포함)
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

    # centroid만 검은 X로 표시
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
    plt.savefig(f"{save_prefix}_centroids.png", dpi=200)
    plt.close()

    print(f"[Saved] {save_prefix}_cluster.png")
    if lengths is not None:
        print(f"[Saved] {save_prefix}_length.png")
    print(f"[Saved] {save_prefix}_centroids.png")


import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def plot_elbow(X, k_min=2, k_max=16, save_path="traj_pca_elbow.png"):
    """
    X: (N, D) feature (X_scaled 추천)
    k_min, k_max: 탐색할 K 범위
    """
    Ks = list(range(k_min, k_max + 1))
    inertias = []

    for K in Ks:
        print(f"[Elbow] Fitting KMeans with K={K} ...")
        kmeans = KMeans(n_clusters=K, random_state=0, n_init="auto")
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)   # 각 클러스터 내 제곱합(SSE)

    # --- Plot ---
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

if __name__ == '__main__':
    
    train_dataset = ...
    trajectory_boundaries = ...
    
    features, lengths = build_traj_features(train_dataset, trajectory_boundaries,
                                        obs_key="observations",
                                        obs_slice=None,  # 또는 slice(0, 3)
                                        k_keyframes=20)
    

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features).astype(np.float64)  # float64로

    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=30)
    labels = clusterer.fit_predict(X_scaled)

    # 시각화용 2D PCA
    pca_2d = PCA(n_components=2, random_state=0)
    X_pca = pca_2d.fit_transform(X_scaled)

    # GMM / Spectral용 약간 더 높은 차원의 PCA (예: 20차원)
    pca_gmm = PCA(n_components=min(20, X_scaled.shape[1]), random_state=0)
    X_gmm = pca_gmm.fit_transform(X_scaled)


    # features, lengths, X_scaled까지 만든 뒤
    # Ks, inertias = plot_elbow(X_scaled, k_min=2, k_max=20, save_path="traj_elbow.png")
    # print("[DEBUG inertials] inertias :", inertias)

    # K=7
    # kmeans = KMeans(n_clusters=K, random_state=0)
    # cluster_ids = kmeans.fit_predict(X_scaled) 

    # PCA + 클러스터 결과 시각화 및 저장
    visualize_traj_clusters(
        X_pca, 
        labels, 
        lengths=lengths, 
        save_prefix=f"traj_pca_hdbscan"
    )
    print(f"hdbsan pngs are saved!")
    # ----------------------------
    #  추가 1) GMM clustering
    # ----------------------------
    K = 7  # elbow 등으로 고른 K 사용
    gmm = GaussianMixture(
        n_components=K,
        covariance_type="full",
        random_state=0,
    )
    gmm_labels = gmm.fit_predict(X_scaled)

    visualize_traj_clusters(
        X_pca,
        gmm_labels,
        lengths=lengths,
        save_prefix="traj_pca_gmm_k7",
        noise_label=-1,   # GMM에는 noise 없음 → 그냥 -1로 두면 자동 무시
    )
    print("GMM pngs are saved!")

    # ----------------------------
    #  추가 2) Spectral clustering
    # ----------------------------
    spectral = SpectralClustering(
        n_clusters=K,
        assign_labels="kmeans",
        affinity="nearest_neighbors",  # trajectory feature라서 NN 기반이 무난
        n_neighbors=10,                # 필요하면 조정
        random_state=0,
    )
    spectral_labels = spectral.fit_predict(X_scaled)

    visualize_traj_clusters(
        X_pca,
        spectral_labels,
        lengths=lengths,
        save_prefix="traj_pca_spectral_k7",
        noise_label=-1,   # noise 개념 없음
    )
    print("Spectral pngs are saved!")
