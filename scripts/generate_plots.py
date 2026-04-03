"""Generate publication-ready figures from saved experiment artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

mpl_config_dir = PROJECT_ROOT / ".cache" / "matplotlib"
mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

import matplotlib

matplotlib.use("Agg")

from src.visualization.plot_alpha_tradeoff import generate_alpha_tradeoff_plot
from src.visualization.plot_hitsk import generate_hits_at_k_plot
from src.visualization.plot_interpretability import (
    generate_metapath_contributions_plot,
    generate_performance_vs_interpretability_plot,
)
from src.visualization.plot_metrics import generate_model_comparison_bar
from src.visualization.plot_pr_curve import generate_pr_curve_comparison
from src.visualization.plot_ranking_distribution import generate_ranking_distribution_plot
from src.visualization.utils import DEFAULT_FIGURE_DIR, DEFAULT_RUN_DIR, ensure_dir

REQUIRED_FIGURES = [
    "model_comparison_bar.png",
    "pr_curve_comparison.png",
    "alpha_tradeoff.png",
    "hits_at_k.png",
    "ranking_distribution.png",
    "metapath_contributions.png",
    "performance_vs_interpretability.png",
]


def parse_args() -> argparse.Namespace:
    """Parse script arguments."""
    parser = argparse.ArgumentParser(description="Generate visualization figures from saved run artifacts.")
    parser.add_argument(
        "--result-dir",
        type=str,
        default=str(DEFAULT_RUN_DIR),
        help="Path to experiment result directory containing saved artifacts.",
    )
    parser.add_argument(
        "--figure-dir",
        type=str,
        default=str(DEFAULT_FIGURE_DIR),
        help="Output directory for generated figures.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing figure files if they already exist.",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Disable PDF export (PNG is always saved).",
    )
    return parser.parse_args()


def _scan_existing_figures(figure_dir: Path) -> None:
    """Print existing figure files before generation starts."""
    ensure_dir(figure_dir)
    existing = sorted(path.name for path in figure_dir.glob("*.png"))
    if not existing:
        print(f"[scan] No existing PNG figures found in: {figure_dir}")
        return
    print(f"[scan] Existing PNG figures in {figure_dir}:")
    for name in existing:
        print(f"  - {name}")


def main() -> None:
    """Run all visualization generators sequentially."""
    args = parse_args()
    result_dir = Path(args.result_dir)
    figure_dir = Path(args.figure_dir)
    save_pdf = not args.no_pdf

    if not result_dir.exists():
        raise FileNotFoundError(f"Result directory does not exist: {result_dir}")

    _scan_existing_figures(figure_dir)

    jobs = [
        ("model_comparison_bar", generate_model_comparison_bar),
        ("pr_curve_comparison", generate_pr_curve_comparison),
        ("alpha_tradeoff", generate_alpha_tradeoff_plot),
        ("hits_at_k", generate_hits_at_k_plot),
        ("ranking_distribution", generate_ranking_distribution_plot),
        ("metapath_contributions", generate_metapath_contributions_plot),
        ("performance_vs_interpretability", generate_performance_vs_interpretability_plot),
    ]

    print("[run] Generating figures...")
    for name, fn in jobs:
        saved = fn(
            result_dir=result_dir,
            figure_dir=figure_dir,
            overwrite=bool(args.overwrite),
            save_pdf=save_pdf,
        )
        status = "saved" if saved else "skipped"
        print(f"  - {name}: {status}")

    print("[done] Figure generation complete.")
    print("[done] Expected core/advanced PNG outputs:")
    for filename in REQUIRED_FIGURES:
        path = figure_dir / filename
        print(f"  - {path}: {'OK' if path.exists() else 'MISSING'}")


if __name__ == "__main__":
    main()
