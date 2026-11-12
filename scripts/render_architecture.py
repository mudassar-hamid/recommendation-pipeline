"""Render the recommendation pipeline architecture as a static PNG."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "architecture.png"
fig, ax = plt.subplots(figsize=(16, 11))
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.axis("off")


def node(x, y, w, h, title, subtitle="", fc="#EAF2F8", ec="#35678A", title_size=9.5):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.04,rounding_size=0.12",
                           linewidth=1.5, edgecolor=ec, facecolor=fc)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center",
            fontsize=title_size, weight="bold")
    if subtitle:
        ax.text(x + w / 2, y + h * 0.28, subtitle, ha="center", va="center",
                fontsize=7.7, wrap=True)


def arrow(x1, y1, x2, y2, color="#526777", style="-|>", lw=1.35, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=12, linewidth=lw,
                                 color=color, linestyle=ls))

# Dashed frame represents Prefect's orchestration boundary.
frame = FancyBboxPatch((0.35, 0.35), 15.3, 10.25,
                       boxstyle="round,pad=0.03,rounding_size=0.15",
                       linewidth=1.7, edgecolor="#6B7280", facecolor="none",
                       linestyle="--")
ax.add_patch(frame)
ax.text(8, 10.78, "Prefect flow - dependencies, parallel model tasks, retries and run state",
        ha="center", fontsize=12.5, weight="bold")

# Main data path.
node(5.6, 9.65, 4.8, 0.65, "MovieLens 1M source", "users.dat | movies.dat | ratings.dat", "#F5F5F5", "#777777")
node(5.6, 8.55, 4.8, 0.68, "1. Data ingestion", "download, extract and normalize to CSV")
node(5.6, 7.40, 4.8, 0.72, "2. Data validation", "missing, duplicates, types, ratings, timestamps and FK checks")
node(0.8, 7.40, 3.6, 0.72, "Error quarantine", "invalid rows + validation report", "#FCE8E6", "#B94A48")
node(5.6, 6.25, 4.8, 0.68, "3. Data preparation", "cleaning + chronological train/test split")
node(5.6, 5.10, 4.8, 0.68, "4. Data transformation", "user/item summaries + sparse interaction matrix")
node(5.6, 3.95, 4.8, 0.68, "5. Feature management", "reusable user, item and interaction features")
node(5.6, 2.85, 4.8, 0.68, "Initialize MLflow experiment", "shared tracking experiment before training", "#F1EAFE", "#7652A3")

# Parallel training branches and evaluation join.
node(0.9, 1.55, 4.5, 0.82, "6A. Popularity baseline", "rank items by positive counts, activity and mean rating", "#EAF6EA", "#3B7A57")
node(10.6, 1.55, 4.5, 0.82, "6B. Personalized recommender", "item-item collaborative filtering with cosine similarity", "#EAF6EA", "#3B7A57")
node(5.6, 0.55, 4.8, 0.66, "7. Evaluation and reporting", "Precision@K | Recall@K | NDCG@K | Coverage | Diversity", "#FFF5DF", "#B07A2A")

# MLflow tracking store is shown separately from the processing path.
node(12.1, 5.35, 2.7, 1.05, "MLflow tracking store", "parameters | metrics\nmodels and reports", "#F1EAFE", "#7652A3")

# Sequential data dependencies.
arrow(8, 9.65, 8, 9.23)
arrow(8, 8.55, 8, 8.12)
arrow(5.6, 7.76, 4.4, 7.76, color="#B94A48")
arrow(8, 7.40, 8, 6.93)
arrow(8, 6.25, 8, 5.78)
arrow(8, 5.10, 8, 4.63)
arrow(8, 3.95, 8, 3.53)

# Fan out to two independent model-training tasks.
arrow(6.8, 2.85, 3.15, 2.37)
arrow(9.2, 2.85, 12.85, 2.37)
arrow(3.15, 1.55, 6.55, 1.21)
arrow(12.85, 1.55, 9.45, 1.21)

# MLflow receives training parameters/artifacts and evaluation outputs.
arrow(10.4, 3.18, 12.1, 5.35, color="#7652A3", ls="--")
arrow(5.4, 1.95, 12.1, 5.55, color="#7652A3", ls="--", lw=1.0)
arrow(10.6, 1.95, 12.1, 5.55, color="#7652A3", ls="--", lw=1.0)
arrow(10.4, 0.88, 12.1, 5.35, color="#7652A3", ls="--", lw=1.0)

ax.text(8, 0.12,
        "All aggregate features, popularity statistics and model fitting use training-period interactions only.",
        ha="center", fontsize=7.8, color="#4B5563")
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {OUT}")
