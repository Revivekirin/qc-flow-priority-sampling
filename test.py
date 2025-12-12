import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# === 1) Trajectory-level feature 추출 ===

K_KEY = 20          # downsample할 keyframe 개수
features = []
lengths = []

data = np.load("traj_features.npz")
print(data.keys)

# for (s, e) in trajectory_boundaries:
#     T = e - s + 1
#     lengths.append(T)

#     # (1) 균일하게 K_KEY개 index 뽑기
#     idx = np.linspace(s, e, K_KEY, dtype=int)
#     pts = ee_pos[idx]      # (K_KEY, 3)

#     # (2) 속도 norm 통계
#     v = np.diff(pts, axis=0)          # (K_KEY-1, 3)
#     v_norm = np.linalg.norm(v, axis=-1)
#     v_mean = v_norm.mean()
#     v_std  = v_norm.std()

#     # (3) 길이 정규화용 임시 placeholder (나중에 scaler가 처리)
#     feat = np.concatenate([
#         np.array([T, v_mean, v_std]),   # 3개
#         pts.flatten()                   # 20 * 3 개
#     ])
#     features.append(feat)

# features = np.stack(features)   # (num_traj, D)
# lengths = np.array(lengths)
# print("Feature shape:", features.shape)
# print("Length stats:", lengths.min(), lengths.max(), lengths.mean())
