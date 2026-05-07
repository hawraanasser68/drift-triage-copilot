"""
Drift detection helpers.
  - PSI  (Population Stability Index) for numeric features and output scores
  - chi² for categorical features
"""

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from model_service.schemas import DriftFeatureDetail, DriftReport

# PSI severity thresholds (industry standard)
PSI_WARNING  = 0.1
PSI_CRITICAL = 0.2

# chi² significance level
CHI2_ALPHA = 0.05


# ---------------------------------------------------------------------------
# PSI
# ---------------------------------------------------------------------------
def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Compute PSI between a reference and a current distribution."""
    # Build bin edges from reference, then apply to current
    breakpoints = np.histogram_bin_edges(reference, bins=bins)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)


    #converting to percentages
    # Replace zeros to avoid log(0)
    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / len(reference))
    cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / len(current))

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


# ---------------------------------------------------------------------------
# chi²
# ---------------------------------------------------------------------------
def _chi2(reference: pd.Series, current: pd.Series) -> tuple[float, float]:
    """Return (chi2_statistic, p_value) for two categorical distributions."""
    all_categories = set(reference.unique()) | set(current.unique())
    ref_counts = reference.value_counts().reindex(all_categories, fill_value=0)
    cur_counts = current.value_counts().reindex(all_categories, fill_value=0)

    contingency = np.array([ref_counts.values, cur_counts.values])
    stat, p_value, _, _ = chi2_contingency(contingency)
    return float(stat), float(p_value)


# ---------------------------------------------------------------------------
# Build a full drift report
# ---------------------------------------------------------------------------
def compute_drift_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    reference_scores: np.ndarray,
    current_scores: np.ndarray,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> DriftReport:
    numeric_details: list[DriftFeatureDetail] = []
    categorical_details: list[DriftFeatureDetail] = []
    worst_severity = "none"

    def _update_severity(psi_val: float) -> None:
        nonlocal worst_severity
        if psi_val >= PSI_CRITICAL:
            worst_severity = "critical"
        elif psi_val >= PSI_WARNING and worst_severity != "critical":
            worst_severity = "warning"

    # Numeric features — PSI
    for col in numeric_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        psi_val = _psi(reference_df[col].values, current_df[col].values)
        drifted = psi_val >= PSI_WARNING
        _update_severity(psi_val)
        numeric_details.append(DriftFeatureDetail(
            feature=col,
            statistic=round(psi_val, 4),
            p_value=None,
            drifted=drifted,
        ))

    # Categorical features — chi²
    for col in categorical_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        stat, p_val = _chi2(reference_df[col], current_df[col])
        drifted = p_val < CHI2_ALPHA
        if drifted and worst_severity == "none":
            worst_severity = "warning"
        categorical_details.append(DriftFeatureDetail(
            feature=col,
            statistic=round(stat, 4),
            p_value=round(p_val, 4),
            drifted=drifted,
        ))

    # Output score distribution — PSI
    output_psi = _psi(reference_scores, current_scores)
    _update_severity(output_psi)

    return DriftReport(
        severity=worst_severity,
        window_size=len(current_df),
        numeric_drift=numeric_details,
        categorical_drift=categorical_details,
        output_drift=round(output_psi, 4),
    )
