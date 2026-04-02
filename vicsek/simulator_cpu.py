"""
Vicsek Model — CPU 물리 엔진

변경사항 (리뷰 반영):
  - step_auto(): N 기반 자동 라우팅 (#2)
  - CELL_THRESHOLD 클래스 변수화 (#3)
  - knn → topologic 용어 (#11)
"""
from __future__ import annotations

import numpy as np

from .config import SimConfig
from .convergence import compute_n_rings
from .cell_search_cpu import topologic_cell_search, metric_cell_search


class VicsekSimulatorCPU:
    """CPU(NumPy) 기반 Vicsek 모델 물리 엔진."""

    CELL_THRESHOLD = 2048  # (#3) N ≥ 이 값이면 셀 리스트 사용

    def __init__(self, cfg: SimConfig, silent: bool = False):
        self.cfg = cfg
        if not silent:
            import platform, os as _os
            cpu = platform.processor() or _os.uname().machine
            print(f"CPU: {cpu}  |  dtype: {cfg.dtype.__name__}")

    # ------------------------------------------------------------------
    # 자동 라우팅 (#2): Simulator가 알고리즘 선택을 책임짐
    # ------------------------------------------------------------------

    def step_auto(
        self, model: str, pos, theta, noise, N: int, fov_rad: float,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """모델과 N 크기에 따라 최적 알고리즘을 자동 선택한다."""
        if model == "metric":
            if N >= self.CELL_THRESHOLD:
                return self.step_metric_cell(pos, theta, noise, N, fov_rad)
            return self.step_metric(pos, theta, noise, N, fov_rad)
        else:  # topologic
            k = min(self.cfg.k_topologic, N - 1) if N > 1 else 0
            if N >= self.CELL_THRESHOLD:
                rho = N / (self.cfg.L ** 2)
                n_rings = compute_n_rings(k, rho, fov_rad, self.cfg._two_pi)
                return self.step_topologic_cell(
                    pos, theta, noise, N, fov_rad, k, n_rings, 1.0
                )
            return self.step_topologic(pos, theta, noise, N, fov_rad, k)

    # ------------------------------------------------------------------
    # Metric 스텝
    # ------------------------------------------------------------------

    def step_metric(self, pos, theta, noise, N, fov_rad):
        cfg = self.cfg
        cos_t = np.cos(theta); sin_t = np.sin(theta)
        dx, dy = self._pbc_displacement(pos)
        dist_sq = dx * dx + dy * dy

        if fov_rad >= cfg._two_pi - cfg._eps_fov or N <= 1:
            nb = (dist_sq < cfg._r_sq) & (dist_sq > cfg._eps_sq)
            np.fill_diagonal(nb, False)
        else:
            nb = self._metric_fov_nb(cos_t, sin_t, dx, dy, dist_sq, fov_rad)

        avg = self._weighted_avg(theta, cos_t, sin_t, nb, cfg.dtype)
        theta[:] = avg + noise
        return (*self._move_and_phi(pos, theta, N), )

    # ------------------------------------------------------------------
    # Topologic (k-NN) 스텝
    # ------------------------------------------------------------------

    def step_topologic(self, pos, theta, noise, N, fov_rad, k):
        cfg = self.cfg
        cos_t = np.cos(theta); sin_t = np.sin(theta)
        dx, dy = self._pbc_displacement(pos)
        dist_sq = dx * dx + dy * dy

        np.fill_diagonal(dist_sq, np.inf)
        if fov_rad < cfg._two_pi - cfg._eps_fov and N > 1:
            dist_sq = self._apply_fov(cos_t, sin_t, dx, dy, dist_sq, fov_rad)

        part = np.argpartition(dist_sq, k, axis=1)[:, :k]
        chosen_d = np.take_along_axis(dist_sq, part, axis=1)
        valid = chosen_d != np.inf
        avg = self._topologic_avg(theta, cos_t, sin_t, part, valid, cfg.dtype)
        theta[:] = avg + noise
        return (*self._move_and_phi(pos, theta, N), )

    # ------------------------------------------------------------------
    # 셀 리스트 기반 스텝
    # ------------------------------------------------------------------

    def step_metric_cell(self, pos, theta, noise, N, fov_rad):
        cfg = self.cfg
        cos_t = np.cos(theta); sin_t = np.sin(theta)
        use_fov = fov_rad < cfg._two_pi - cfg._eps_fov and N > 1
        cos_half = cfg.get_cos_half(fov_rad)

        n_cells_1d = max(1, int(cfg.L / cfg.r_metric))
        cs = cfg.L / n_cells_1d
        sort_idx, cell_start = self._build_cell_list(pos, n_cells_1d, cs, N)
        px = np.ascontiguousarray(pos[:, 0])
        py = np.ascontiguousarray(pos[:, 1])

        avg_cos, avg_sin = metric_cell_search(
            px, py, cos_t, sin_t, sort_idx, cell_start,
            N, n_cells_1d, cs, cfg.L, cfg._r_sq, use_fov, cos_half,
        )
        theta[:] = np.arctan2(avg_sin, avg_cos) + noise
        return (*self._move_and_phi(pos, theta, N), )

    def step_topologic_cell(self, pos, theta, noise, N, fov_rad, k, n_rings, cell_size):
        cfg = self.cfg
        cos_t = np.cos(theta); sin_t = np.sin(theta)
        use_fov = fov_rad < cfg._two_pi - cfg._eps_fov and N > 1
        cos_half = cfg.get_cos_half(fov_rad)

        n_cells_1d = max(1, int(cfg.L / cell_size))
        cs = cfg.L / n_cells_1d
        sort_idx, cell_start = self._build_cell_list(pos, n_cells_1d, cs, N)
        px = np.ascontiguousarray(pos[:, 0])
        py = np.ascontiguousarray(pos[:, 1])

        avg_cos, avg_sin = topologic_cell_search(
            px, py, cos_t, sin_t, theta, sort_idx, cell_start,
            N, n_cells_1d, cs, cfg.L, k, n_rings, use_fov, cos_half,
        )
        theta[:] = np.arctan2(avg_sin, avg_cos) + noise
        return (*self._move_and_phi(pos, theta, N), )

    # ------------------------------------------------------------------
    # 내부 헬퍼
    # ------------------------------------------------------------------

    def _pbc_displacement(self, pos):
        L = self.cfg.L
        dx = pos[:, 0:1] - pos[:, 0]
        dy = pos[:, 1:2] - pos[:, 1]
        dx -= L * np.round(dx / L); dy -= L * np.round(dy / L)
        return dx, dy

    def _metric_fov_nb(self, cos_t, sin_t, dx, dy, dist_sq, fov_rad):
        cfg = self.cfg
        norm = np.sqrt(np.maximum(dist_sq, cfg._eps_sq))
        dot = np.clip(cos_t[:, None] * (dx / norm) + sin_t[:, None] * (dy / norm), -1.0, 1.0)
        fov_ok = dot >= cfg.get_cos_half(fov_rad)
        np.fill_diagonal(fov_ok, False)
        return (dist_sq < cfg._r_sq) & (dist_sq > cfg._eps_sq) & fov_ok

    def _apply_fov(self, cos_t, sin_t, dx, dy, dist_sq, fov_rad):
        cfg = self.cfg
        norm = np.sqrt(np.maximum(dist_sq, cfg._eps_sq))
        dot = np.clip(cos_t[:, None] * (dx / norm) + sin_t[:, None] * (dy / norm), -1.0, 1.0)
        return np.where(dot >= cfg.get_cos_half(fov_rad), dist_sq, np.inf)

    @staticmethod
    def _weighted_avg(theta, cos_t, sin_t, mask, dtype):
        n = mask.sum(axis=1).astype(dtype)
        sc = np.where(mask, cos_t, 0.0).sum(axis=1)
        ss = np.where(mask, sin_t, 0.0).sum(axis=1)
        s = np.where(n > 0, n, 1.0)
        return np.where(n > 0, np.arctan2(ss / s, sc / s), theta)

    @staticmethod
    def _topologic_avg(theta, cos_t, sin_t, part, valid, dtype):
        sc = np.where(valid, cos_t[part], 0.0).sum(axis=1)
        ss = np.where(valid, sin_t[part], 0.0).sum(axis=1)
        n = valid.sum(axis=1).astype(dtype)
        s = np.where(n > 0, n, 1.0)
        return np.where(n > 0, np.arctan2(ss / s, sc / s), theta)

    def _move_and_phi(self, pos, theta, N):
        cos_new = np.cos(theta); sin_new = np.sin(theta)
        L, vdt = self.cfg.L, self.cfg._vdt
        pos[:, 0] = (pos[:, 0] + vdt * cos_new) % L
        pos[:, 1] = (pos[:, 1] + vdt * sin_new) % L
        phi = float(np.sqrt(cos_new.sum() ** 2 + sin_new.sum() ** 2)) / N
        return pos, theta, phi

    @staticmethod
    def _build_cell_list(pos, n_cells_1d, cs, N):
        cx = np.minimum((pos[:, 0] / cs).astype(np.int32), n_cells_1d - 1)
        cy = np.minimum((pos[:, 1] / cs).astype(np.int32), n_cells_1d - 1)
        cell_id = cy * n_cells_1d + cx
        sort_idx = np.argsort(cell_id)
        sorted_cid = cell_id[sort_idx]
        n_cells = n_cells_1d * n_cells_1d
        cell_start = np.full(n_cells + 1, N, dtype=np.int32)
        bounds = np.where(
            np.diff(np.concatenate([np.array([-1], dtype=np.int32), sorted_cid])) != 0
        )[0]
        cell_start[sorted_cid[bounds]] = bounds.astype(np.int32)
        for c in range(n_cells - 1, -1, -1):
            if cell_start[c] == N:
                cell_start[c] = cell_start[c + 1]
        return sort_idx, cell_start