"""
Vicsek Model — 통합 Trial 실행기

변경사항 (리뷰 반영):
  - CPU/GPU 코드 통합: xp(배열 모듈) 동적 주입 (#1)
  - step_auto() 사용으로 알고리즘 선택 위임 (#2)
  - 노이즈: η × uniform(-π, π) (#5)
"""
from __future__ import annotations

import numpy as np

from .config import SimConfig
from .convergence import check_converged


class TrialRunner:
    """CPU/GPU 통합 단일 trial 실행기.

    Args:
        sim: VicsekSimulatorCPU 또는 VicsekSimulatorGPU 인스턴스.
             step_auto(model, pos, theta, noise, N, fov_rad) 인터페이스 필수.
        xp:  배열 모듈 (numpy 또는 cupy).
    """

    def __init__(self, sim, xp=None):
        self.sim = sim
        self.xp = xp if xp is not None else np

    def run(
        self,
        N: int,
        fov_rad: float,
        eta: float,
        models: list[str] | None = None,
    ) -> dict[str, np.ndarray]:
        """단일 trial을 실행하고 model별 측정 윈도우 φ 시계열을 반환.

        Args:
            eta: [0, 1] 정규화 노이즈 강도. 실제 노이즈 = η × uniform(-π, π)
        """
        xp = self.xp
        cfg = self.sim.cfg
        extra = cfg.conv_extra_steps
        chunk = cfg.noise_chunk

        if models is None:
            models = cfg.active_models()

        pos_init = xp.random.uniform(0.0, cfg.L, (N, 2)).astype(cfg.dtype)
        theta_init = xp.random.uniform(-np.pi, np.pi, (N,)).astype(cfg.dtype)

        # ── 노이즈 청크 생성 (#5): η × uniform(-π, π) ──────────────
        def _new_chunk():
            return {
                m: (
                    eta
                    * xp.random.uniform(
                        -np.pi, np.pi, (chunk, N)
                    ).astype(cfg.dtype)
                )
                for m in models
            }

        noise_buf = _new_chunk()
        buf_pos = 0

        pos = {m: pos_init.copy() for m in models}
        theta = {m: theta_init.copy() for m in models}

        hist_buf_size = cfg.conv_min_steps + cfg.conv_window * 2
        hist = {m: np.zeros(hist_buf_size, dtype=np.float32) for m in models}

        # ── 메인 루프 ─────────────────────────────────────────────
        for t in range(cfg.max_steps):
            if buf_pos >= chunk:
                noise_buf = _new_chunk()
                buf_pos = 0

            for m in models:
                noise = noise_buf[m][buf_pos]
                # step_auto가 모델/N 크기에 따라 최적 알고리즘 선택 (#2)
                pos[m], theta[m], phi = self.sim.step_auto(
                    m, pos[m], theta[m], noise, N, fov_rad,
                )
                hist[m][t % hist_buf_size] = phi

            buf_pos += 1

            if t >= cfg.conv_min_steps and t % cfg._check_interval == 0:
                if all(check_converged(hist[m], t, cfg) for m in models):
                    break

        # ── 측정 윈도우 ───────────────────────────────────────────
        meas = {m: np.empty(extra, dtype=np.float32) for m in models}
        for i in range(extra):
            if buf_pos >= chunk:
                noise_buf = _new_chunk()
                buf_pos = 0
            for m in models:
                noise = noise_buf[m][buf_pos]
                pos[m], theta[m], phi = self.sim.step_auto(
                    m, pos[m], theta[m], noise, N, fov_rad,
                )
                meas[m][i] = phi
            buf_pos += 1

        return meas