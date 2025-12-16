import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

TASKS = ["lift", "can", "square", "transport"]
METHODS = ["kmeans", "hdbscan", "gmm", "spectral"]

def safe_load_image(path):
    """이미지 파일을 로드. 없으면 None 반환."""
    if path is None:
        return None
    if not os.path.exists(path):
        return None
    return mpimg.imread(path)

def make_5x4_grid(
    elbow_paths_by_task,
    vis_paths_by_method_and_task,
    out_path="kmeans_cluster_grid_5x4.png",
    dpi=200,
    figsize=(18, 20),
):
    """
    elbow_paths_by_task: dict[task] -> path (1행)
    vis_paths_by_method_and_task: dict[method][task] -> path (2~5행)
    """

    fig, axes = plt.subplots(nrows=5, ncols=4, figsize=figsize)

    # ---- Column titles (task)
    for c, task in enumerate(TASKS):
        axes[0, c].set_title(task, fontsize=14, pad=10)

    # ---- Row labels
    row_labels = ["Elbow (KMeans)", "KMeans", "HDBSCAN", "GMM", "Spectral"]
    for r in range(5):
        axes[r, 0].set_ylabel(row_labels[r], fontsize=12, rotation=90, labelpad=10)

    # ---- 1st row: elbow plots
    for c, task in enumerate(TASKS):
        ax = axes[0, c]
        img = safe_load_image(elbow_paths_by_task.get(task))
        if img is None:
            ax.text(0.5, 0.5, f"Missing\n{task} elbow", ha="center", va="center", fontsize=10)
        else:
            ax.imshow(img)
        ax.axis("off")

    # ---- Rows 2~5: method visualizations
    method_to_row = {"kmeans": 1, "hdbscan": 2, "gmm": 3, "spectral": 4}
    for method in METHODS:
        r = method_to_row[method]
        for c, task in enumerate(TASKS):
            ax = axes[r, c]
            img = safe_load_image(vis_paths_by_method_and_task.get(method, {}).get(task))
            if img is None:
                ax.text(0.5, 0.5, f"Missing\n{method}-{task}", ha="center", va="center", fontsize=10)
            else:
                ax.imshow(img)
            ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_path}")

if __name__ == "__main__":
    # ✅ 당신이 저장해둔 파일 경로를 여기에 맞춰 넣기
    elbow_paths_by_task = {
        "lift":      "plot/lift_elbow.png",
        "can":       "plot/can_elbow.png",
        "square":    "plot/square_elbow.png",
        "transport": "plot/transport_elbow.png",
    }

    vis_paths_by_method_and_task = {
        "kmeans": {
            "lift":      "plot/lift_kmeans.png",
            "can":       "plot/can_kmeans.png",
            "square":    "plot/square_kmeans.png",
            "transport": "plot/transport_kmeans.png",
        },
        "hdbscan": {
            "lift":      "plot/lift_hdbscan.png",
            "can":       "plot/can_hdbscan.png",
            "square":    "plot/square_hdbscan.png",
            "transport": "plot/transport_hdbscan.png",
        },
        "gmm": {
            "lift":      "plot/lift_gmm.png",
            "can":       "plot/can_gmm.png",
            "square":    "plot/square_gmm.png",
            "transport": "plot/transport_gmm.png",
        },
        "spectral": {
            "lift":      "plot/lift_spectral.png",
            "can":       "plot/can_spectral.png",
            "square":    "plot/square_spectral.png",
            "transport": "plot/transport_spectral.png",
        },
    }

    make_5x4_grid(
        elbow_paths_by_task=elbow_paths_by_task,
        vis_paths_by_method_and_task=vis_paths_by_method_and_task,
        out_path="cluster_viz_5x4.png",
        dpi=250,
        figsize=(18, 22),
    )
