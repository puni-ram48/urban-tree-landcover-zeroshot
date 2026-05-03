#!/usr/bin/env python3
"""
Statistical Significance Testing — Wilcoxon Signed-Rank Test

Four focused tests addressing each research question:

  RQ1: E1 vs E4 coverage           (segmentation fusion effect)
  RQ2: Prompt 0 vs Prompt 1.3      (aerial prompt design effect)
  RQ3: Zero vs Dual Composite      (crop method effect)
  RQ4: Aerial 2020 vs Google Sat   (cross-dataset generalisation)

Test: Wilcoxon signed-rank (non-parametric, paired, two-sided)
Significance level: alpha = 0.05
Sample size: 95 tree locations

Usage:
  python statistical_tests.py
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# PATHS — update to match your setup
BASE = "<output_folder>/evaluation/outputs"

# RQ1 — Segmentation coverage
PATH_E1 = "segmentation/outputs/exp01_single_default/metrics/per_image_metrics.csv"
PATH_E4 = "segmentation/outputs/exp04_multiscale_finetuned/metrics/per_image_metrics.csv"

# RQ2 — Prompt design
PATH_PROMPT0 = f"{BASE}/aerial2020_zero_prompt0_results/per_tree_metrics_pixelgt.csv"
PATH_PROMPT13 = f"{BASE}/aerial2020_zero_prompt1.3_results/per_tree_metrics_pixelgt.csv"

# RQ3 — Crop method
PATH_ZERO = f"{BASE}/aerial2020_zero_prompt1.3_results/per_tree_metrics_pixelgt.csv"
PATH_DUAL = f"{BASE}/aerial2020_dual_composite_prompt1.3_results/per_tree_metrics_pixelgt.csv"

# RQ4 — Cross-dataset
PATH_AERIAL2020 = f"{BASE}/aerial2020_zero_prompt1.3_results/per_tree_metrics_pixelgt.csv"
PATH_GOOGLE = f"{BASE}/googlesatellite_zero_prompt1.3_results/per_tree_metrics_pixelgt.csv"


# CONFIGURATION
ALPHA = 0.05

# HELPERS
def run_test(a, b, label_a, label_b, metric):
    """Wilcoxon signed-rank test between two paired arrays."""
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    n = len(a)

    if n < 10:
        return {
            "Comparison": f"{label_a} vs {label_b}",
            "Metric": metric,
            "N": n,
            "Mean A": round(float(np.mean(a)), 4),
            "Mean B": round(float(np.mean(b)), 4),
            "Difference": round(float(np.mean(b) - np.mean(a)), 4),
            "Statistic": None,
            "p-value": None,
            "Significant": "insufficient data",
        }

    stat, p = wilcoxon(a, b, alternative="two-sided")

    return {
        "Comparison": f"{label_a} vs {label_b}",
        "Metric": metric,
        "N": n,
        "Mean A": round(float(np.mean(a)), 4),
        "Mean B": round(float(np.mean(b)), 4),
        "Difference": round(float(np.mean(b) - np.mean(a)), 4),
        "Statistic": round(float(stat), 4),
        "p-value": round(float(p), 4),
        "Significant": f"YES (p={p:.4f})" if p < ALPHA else f"NO (p={p:.4f})",
    }


def print_section(title):
    """Print section header."""
    print("\n" + "=" * 90)
    print(f"  {title}")
    print("=" * 90)


def print_result(result):
    """Print single test result."""
    print(f"\n  Comparison : {result['Comparison']}")
    print(f"  Metric     : {result['Metric']}")
    print(f"  N          : {result['N']} trees")
    print(f"  Mean A     : {result['Mean A']}")
    print(f"  Mean B     : {result['Mean B']}")
    print(f"  Difference : {result['Difference']} (B - A)")
    if result['Statistic'] is not None:
        print(f"  Statistic  : {result['Statistic']}")
        print(f"  p-value    : {result['p-value']}")
    print(f"  Significant: {result['Significant']}")


# RQ1 — E1 vs E4: Coverage
print_section("RQ1: Does multi-scale fusion improve coverage? (E1 vs E4)")

result_rq1 = None
try:
    e1 = pd.read_csv(PATH_E1)
    e4 = pd.read_csv(PATH_E4)

    result_rq1 = run_test(
        e1["pixel_coverage_percent"].values,
        e4["pixel_coverage_percent"].values,
        "E1 (single-scale)",
        "E4 (multi-scale)",
        "Pixel Coverage (%)"
    )
    print_result(result_rq1)

except FileNotFoundError as e:
    print(f"\n  ERROR: {e}")


# RQ2 — Prompt 0 vs Prompt 1.3
print_section("RQ2: Does aerial-specific prompt improve classification? (P0 vs P1.3)")

rq2_results = []
try:
    p0 = pd.read_csv(PATH_PROMPT0)
    p13 = pd.read_csv(PATH_PROMPT13)

    p0 = p0.sort_values("dev_eui").reset_index(drop=True)
    p13 = p13.sort_values("dev_eui").reset_index(drop=True)

    rq2_tests = [
        ("vegetation_2m5_f1", "Vegetation F1 Ring 1"),
        ("vegetation_5m0_f1", "Vegetation F1 Ring 2"),
        ("vegetation_7m5_f1", "Vegetation F1 Ring 3"),
        ("road_2m5_f1", "Road F1 Ring 1"),
        ("road_5m0_f1", "Road F1 Ring 2"),
        ("road_7m5_f1", "Road F1 Ring 3"),
    ]

    for col, label in rq2_tests:
        r = run_test(
            p0[col].values,
            p13[col].values,
            "Prompt 0 (generic)",
            "Prompt 1.3 (aerial)",
            label
        )
        print_result(r)
        rq2_results.append(r)

except FileNotFoundError as e:
    print(f"\n  ERROR: {e}")

# RQ3 — Zero vs Dual Composite
print_section("RQ3: Does crop method affect accuracy? (Zero vs Dual Composite)")

rq3_results = []
try:
    zero = pd.read_csv(PATH_ZERO)
    dual = pd.read_csv(PATH_DUAL)

    zero = zero.sort_values("dev_eui").reset_index(drop=True)
    dual = dual.sort_values("dev_eui").reset_index(drop=True)

    rq3_tests = [
        ("vegetation_2m5_mae", "Vegetation MAE Ring 1"),
        ("vegetation_5m0_mae", "Vegetation MAE Ring 2"),
        ("vegetation_7m5_mae", "Vegetation MAE Ring 3"),
    ]

    for col, label in rq3_tests:
        r = run_test(
            zero[col].values,
            dual[col].values,
            "Zero crop",
            "Dual Composite",
            label
        )
        print_result(r)
        rq3_results.append(r)

except FileNotFoundError as e:
    print(f"\n  ERROR: {e}")

# RQ4 — Cross-dataset
print_section("RQ4: Does pipeline generalise across datasets? (Aerial 2020 vs Google)")

rq4_results = []
try:
    a2020 = pd.read_csv(PATH_AERIAL2020)
    google = pd.read_csv(PATH_GOOGLE)

    a2020 = a2020.sort_values("dev_eui").reset_index(drop=True)
    google = google.sort_values("dev_eui").reset_index(drop=True)

    rq4_tests = [
        ("vegetation_2m5_f1", "Vegetation F1 Ring 1"),
        ("vegetation_5m0_f1", "Vegetation F1 Ring 2"),
        ("vegetation_7m5_f1", "Vegetation F1 Ring 3"),
        ("vegetation_2m5_mae", "Vegetation MAE Ring 1"),
    ]

    for col, label in rq4_tests:
        r = run_test(
            a2020[col].values,
            google[col].values,
            "Aerial 2020",
            "Google Satellite",
            label
        )
        print_result(r)
        rq4_results.append(r)

except FileNotFoundError as e:
    print(f"\n  ERROR: {e}")

# SAVE RESULTS
all_rows = []
if result_rq1:
    result_rq1["RQ"] = "RQ1"
    all_rows.append(result_rq1)

for r in rq2_results:
    r["RQ"] = "RQ2"
    all_rows.append(r)

for r in rq3_results:
    r["RQ"] = "RQ3"
    all_rows.append(r)

for r in rq4_results:
    r["RQ"] = "RQ4"
    all_rows.append(r)

if all_rows:
    df_all = pd.DataFrame(all_rows)
    output_path = f"{BASE}/statistical_tests_results.csv"
    df_all.to_csv(output_path, index=False)

    print("\n" + "=" * 90)
    print("  SUMMARY OF ALL TESTS")
    print("=" * 90)
    print(df_all[["RQ", "Comparison", "Metric", "Mean A", "Mean B",
                  "Difference", "p-value", "Significant"]].to_string(index=False))
    print(f"\nSaved: {output_path}")

print("\n" + "=" * 90)
print("  DONE")
print("=" * 90)
