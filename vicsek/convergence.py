"""
Vicsek Model — 수렴 판정 유틸리티

링 버퍼 기반 수렴 검사 + Poisson 역함수 기반 셀 링 수 추정.
"""
from __future__ import annotations

import math

import numpy as np

from .config import SimConfig


def check_converged(hist: np.ndarray, t: int, cfg: SimConfig) -> bool:
    """링 버퍼 hist에서 수렴 여부를 판정한다."""
    w = cfg.conv_window
    size = len(hist)
    idx_prev = np.arange(t - 2 * w, t - w) % size
    idx_curr = np.arange(t - w, t) % size
    prev = hist[idx_prev].mean()
    curr = hist[idx_curr].mean()
    return abs(float(curr) - float(prev)) <= cfg.conv_threshold


def compute_n_rings(
    k: int, rho: float, fov_rad: float, two_pi: float
) -> int:
    """밀도와 FOV에서 k개를 채우기에 충분한 링 수를 추정한다."""
    MIN_FOV_RAD = math.radians(5.0)
    fov_fraction = min(fov_rad / two_pi, 1.0)
    fov_fraction = max(fov_fraction, MIN_FOV_RAD / two_pi)
    p_success = 0.9999

    def poisson_cdf(lam: float, n: int) -> float:
        total = 0.0
        term = math.exp(-lam)
        for i in range(n + 1):
            total += term
            term *= lam / (i + 1)
        return min(total, 1.0)

    lo, hi = float(k) / 10.0, float(k) * 500.0
    for _ in range(80):
        mid = (lo + hi) * 0.5
        prob = 1.0 - poisson_cdf(mid, k - 1)
        if prob >= p_success:
            hi = mid
        else:
            lo = mid

    r = math.sqrt(hi / (rho * math.pi * fov_fraction))
    return max(2, math.ceil(r))