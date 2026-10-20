"""Draw the processing-pipeline diagram (Figure 1 of the report) with matplotlib.

Run from the repository root:  uv run python report/draw_pipeline.py
Writes reports/figures/F01_pipeline.png. Kept outside the mlda package because it is
documentation, not analysis: nothing in the pipeline depends on it.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "F01_pipeline.png"

BOX_W, BOX_H = 3.25, 1.12
FILL_DATA = "#E6F1FB"
FILL_FIT = "#FAEEDA"
FILL_EVAL = "#E1F5EE"
EDGE = "#444441"


def box(ax, x: float, y: float, title: str, lines: list[str], fill: str) -> None:
    patch = FancyBboxPatch((x - BOX_W / 2, y - BOX_H / 2), BOX_W, BOX_H, boxstyle="round,pad=0.02,rounding_size=0.08",
                           linewidth=1.0, edgecolor=EDGE, facecolor=fill)
    ax.add_patch(patch)
    ax.text(x, y + 0.33, title, ha="center", va="center", fontsize=8.3, fontweight="bold", color="#1f2a37")
    ax.text(x, y - 0.14, "\n".join(lines), ha="center", va="center", fontsize=6.7, color="#2c2c2a", linespacing=1.25)


def arrow(ax, x0: float, y0: float, x1: float, y1: float) -> None:
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, linewidth=1.0, color=EDGE))


def draw() -> Path:
    fig, ax = plt.subplots(figsize=(6.3, 4.5))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    c1, c2, c3, c4 = 2.0, 5.65, 9.3, 10.6
    r1, r2, r3, r4 = 6.3, 4.55, 2.45, 0.95

    box(ax, c1, r1, "Three MSHA extracts", ["Accidents 275,220 rows", "Mines 92,060 rows", "Production 658,922 rows"], FILL_DATA)
    box(ax, c2, r1, "Filter and label", ["2010-2025, injury codes 01-06", "lost_time = codes 01-04", "N = 101,609"], FILL_DATA)
    box(ax, c3, r1, "Join and lag", ["Mines on MINE_ID; production", "on MINE_ID and year - 1; cases", "filed before the accident date"], FILL_DATA)
    arrow(ax, c1 + BOX_W / 2, r1, c2 - BOX_W / 2, r1)
    arrow(ax, c2 + BOX_W / 2, r1, c3 - BOX_W / 2, r1)

    box(ax, c3, r2, "Error rules and features", ["30 documented rules", "52 inputs: 20 categorical,", "20 numeric, 12 binary"], FILL_DATA)
    box(ax, c2, r2, "Grouped stratified split", ["StratifiedGroupKFold on", "mine-year groups; fold 0 held out", "(20 %); 5-fold CV on the rest"], FILL_DATA)
    arrow(ax, c3, r1 - BOX_H / 2, c3, r2 + BOX_H / 2)
    arrow(ax, c3 - BOX_W / 2, r2, c2 + BOX_W / 2, r2)

    region = FancyBboxPatch((0.2, 0.25), 7.35, 3.05, boxstyle="round,pad=0.02,rounding_size=0.1", linewidth=1.2,
                            edgecolor="#BA7517", facecolor="none", linestyle=(0, (5, 3)))
    ax.add_patch(region)
    ax.text(0.45, 0.72, "Fitted on the training split only,\ninside one scikit-learn Pipeline", fontsize=7.0,
            color="#854F0B", ha="left", va="center", fontweight="bold")

    box(ax, c1, r3, "ColumnTransformer", ["median imputer, StandardScaler;", "one-hot with infrequent bucket;", "binary flags passed through"], FILL_FIT)
    box(ax, c2, r3, "8 models, GridSearchCV", ["LogReg, KNN, Naive Bayes, SVM,", "tree, random forest, HistGB, MLP;", "refit on average precision"], FILL_FIT)
    box(ax, c2, r4, "Out-of-fold thresholds", ["specificity 0.40 operating point,", "50 % flag rate, cost-optimal", "threshold per cost ratio"], FILL_FIT)
    arrow(ax, c2, r2 - BOX_H / 2, c2, r3 + BOX_H / 2)
    arrow(ax, c1 + BOX_W / 2, r3, c2 - BOX_W / 2, r3)
    arrow(ax, c2, r3 - BOX_H / 2, c2, r4 + BOX_H / 2)

    box(ax, c4, r3, "Held-out test fold", ["TPR, TNR, AUC, PR-AUC, F1,", "Brier, cost per 1,000 cases,", "McNemar, bootstrap CIs"], FILL_EVAL)
    box(ax, c4, r4, "Deployment", ["joblib artefacts, metrics JSON,", "Streamlit app (seven pages),", "Docker image, CI"], FILL_EVAL)
    arrow(ax, c2 + BOX_W / 2, r3, c4 - BOX_W / 2, r3)
    arrow(ax, c2 + BOX_W / 2, r4, c4 - BOX_W / 2, r4)
    arrow(ax, c4, r3 - BOX_H / 2, c4, r4 + BOX_H / 2)

    fig.tight_layout(pad=0.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    return OUT


if __name__ == "__main__":
    print(draw())
