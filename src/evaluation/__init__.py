"""Evaluation and comparison utilities."""

from .metrics import (
    # Core classes
    TrajectoryResult,
    MethodSummary,
    ExtendedMetrics,
    # Core functions
    evaluate_method,
    summarize_results,
    compare_methods,
    save_results,
    print_comparison_table,
    # Extended metrics (AnDi-style)
    compute_extended_metrics,
    breakdown_by_alpha,
    breakdown_by_length,
    # Legacy
    results_by_alpha,
)

__all__ = [
    # Core classes
    "TrajectoryResult",
    "MethodSummary",
    "ExtendedMetrics",
    # Core functions
    "evaluate_method",
    "summarize_results",
    "compare_methods",
    "save_results",
    "print_comparison_table",
    # Extended metrics (AnDi-style)
    "compute_extended_metrics",
    "breakdown_by_alpha",
    "breakdown_by_length",
    # Legacy
    "results_by_alpha",
]







