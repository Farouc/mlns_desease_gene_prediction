"""Visualization package for publication-ready result figures."""

from src.visualization.plot_alpha_tradeoff import generate_alpha_tradeoff_plot
from src.visualization.plot_hitsk import generate_hits_at_k_plot
from src.visualization.plot_interpretability import (
    generate_metapath_contributions_plot,
    generate_performance_vs_interpretability_plot,
)
from src.visualization.plot_metrics import generate_model_comparison_bar
from src.visualization.plot_pr_curve import generate_pr_curve_comparison
from src.visualization.plot_ranking_distribution import generate_ranking_distribution_plot

__all__ = [
    "generate_alpha_tradeoff_plot",
    "generate_hits_at_k_plot",
    "generate_metapath_contributions_plot",
    "generate_model_comparison_bar",
    "generate_performance_vs_interpretability_plot",
    "generate_pr_curve_comparison",
    "generate_ranking_distribution_plot",
]
