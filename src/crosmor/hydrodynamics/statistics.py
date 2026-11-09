"""Wave spectrum statistical estimators matching CROSMOR STAT.F and ORDER.F."""

from typing import Tuple
import numpy as np


def _order_descending(values: np.ndarray, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sort values descending and reorder corresponding probabilities matching ORDER.F."""
    order = np.argsort(-values)
    return values[order], probs[order]


def _top_fraction_mean(values: np.ndarray, probs: np.ndarray, target_frac: float) -> float:
    """Compute the weighted mean of the top target_frac (e.g. 0.33 or 0.10) matching STAT.F."""
    val_sorted, p_sorted = _order_descending(values, probs)
    accum_p = 0.0
    accum_val = 0.0

    for i in range(len(val_sorted)):
        p = p_sorted[i]
        v = val_sorted[i]
        if accum_p + p <= target_frac:
            accum_p += p
            accum_val += p * v
        else:
            remain = target_frac - accum_p
            accum_val += remain * v
            break

    return float(accum_val / target_frac) if target_frac > 0 else 0.0


def compute_wave_statistics(
    probs: np.ndarray,
    periods: np.ndarray,
    heights: np.ndarray,
    uwf: np.ndarray,
    uwb: np.ndarray,
    urr: np.ndarray,
    ubb: np.ndarray,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """Compute H1/3, T1/3, H1/10, T1/10, Uwf1/3, Uwb1/3, Urr1/3, Ubb1/3 matching STAT.F."""
    nh = len(probs)
    if nh == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    if nh == 1:
        return (
            float(heights[0]),
            float(periods[0]),
            float(heights[0]),
            float(periods[0]),
            float(uwf[0]),
            float(uwb[0]),
            float(urr[0]),
            float(ubb[0]),
        )

    # Re-normalize probs just in case
    p_norm = probs / np.sum(probs)

    h10 = _top_fraction_mean(heights, p_norm, 0.10)
    h3 = _top_fraction_mean(heights, p_norm, 0.33)

    t10 = _top_fraction_mean(periods, p_norm, 0.10)
    t3 = _top_fraction_mean(periods, p_norm, 0.33)

    uwf3 = _top_fraction_mean(uwf, p_norm, 0.33)
    uwb3 = _top_fraction_mean(uwb, p_norm, 0.33)

    # In Fortran STAT.F urr3 and ubb3 sum p(i)*urr(i)
    urr3 = float(np.sum(p_norm * urr))
    ubb3 = float(np.sum(p_norm * ubb))

    return h3, t3, h10, t10, uwf3, uwb3, urr3, ubb3
