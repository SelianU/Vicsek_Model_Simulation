"""
Vicsek Model — 통합 설정 (SimConfig)

변경사항 (리뷰 반영):
  - knn → topologic 용어 정립 (#11)
  - η ∈ [0, 1] 정규화, 실제 노이즈 = η × uniform(-π, π) (#5)
  - 통합 출력 디렉토리 Vicsek_Results (#6)
  - v_values 스캔 기능 추가 (#12)
  - CELL_THRESHOLD 상수를 SimConfig에서 관리하지 않음
    (각 Simulator 클래스 변수로 이동, #3)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SimConfig:
    """시뮬레이션에 필요한 모든 파라미터를 관리하는 클래스."""

    # --- 실행할 모델 ---
    run_metric: bool = True
    run_topologic: bool = True  # 구 run_knn (#11)

    # --- 핵심 물리 파라미터 ---
    r_metric: float = 1.0
    k_topologic: int = 5  # 구 k_knn (#11)
    L: float = 20.0
    v: float = 0.1
    dt: float = 1.0

    # --- 밀도 기반 N 자동 계산 ---
    density: float = 0.0
    density_values: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )

    # --- 실행 파라미터 ---
    max_steps: int = 10000
    noise_chunk: int = 1000
    num_trials: int = 100
    num_workers: int = -1

    # --- 수렴 판단 (Early Stopping) ---
    conv_window: int = 300
    conv_threshold: float = 0.01
    conv_min_steps: int = 1000
    conv_extra_steps: int = 300

    # --- 스캔 파라미터 ---
    fov_angles_deg: np.ndarray = field(
        default_factory=lambda: np.array(
            [60, 120, 180, 240, 300, 360], dtype=np.int32
        )
    )
    n_values: np.ndarray = field(
        default_factory=lambda: np.array(
            [40, 100, 200, 400, 800, 1200, 1600], dtype=np.int32
        )
    )
    # η ∈ [0, 1] 정규화: 0=무노이즈, 1=최대 노이즈(-π,π) (#5)
    eta_values: np.ndarray = field(
        default_factory=lambda: np.linspace(0.0, 1.0, 17)
    )
    # v 스캔 값 목록 (#12). 기본값은 단일 값.
    v_values: np.ndarray = field(
        default_factory=lambda: np.array([0.1])
    )

    # --- 출력 ---
    base_out_dir: str = "Vicsek_Results"  # 통합 출력 (#6)
    dtype: type = np.float32

    # --- 파생 상수 ---
    _vdt: float = field(init=False)
    _r_sq: float = field(init=False)
    _two_pi: float = field(init=False)
    _eps_sq: float = field(init=False)
    _eps_fov: float = field(init=False)
    _check_interval: int = field(init=False)
    _cos_half_fov: dict = field(init=False)
    _fov_rad: np.ndarray = field(init=False)

    def __post_init__(self):
        self._vdt = float(self.v * self.dt)
        self._r_sq = float(self.r_metric ** 2)
        self._two_pi = float(2.0 * np.pi)
        self._eps_sq = float(1e-18)
        self._eps_fov = float(1e-6)
        self._check_interval = max(1, self.conv_window // 4)

        if self.num_workers <= 0:
            self.num_workers = os.cpu_count() or 1

        # 밀도 → N 변환
        if len(self.density_values) > 0:
            self.n_values = np.maximum(
                2,
                np.round(self.density_values * self.L ** 2).astype(np.int32),
            )
        elif self.density > 0.0:
            n = max(2, round(self.density * self.L ** 2))
            self.n_values = np.array([n], dtype=np.int32)

        self._refresh_fov_cache()

    def set_velocity(self, v: float) -> None:
        """속도 변경 시 파생 상수 재계산 (#12)."""
        self.v = v
        self._vdt = float(v * self.dt)

    def _refresh_fov_cache(self):
        self._fov_rad = np.deg2rad(self.fov_angles_deg).astype(np.float32)
        self._cos_half_fov = {
            float(r): float(np.cos(r / 2.0)) for r in self._fov_rad
        }

    def get_cos_half(self, fov_rad: float) -> float:
        """_cos_half_fov 캐시 조회, 없으면 계산 후 캐시."""
        val = self._cos_half_fov.get(fov_rad)
        if val is None:
            val = float(np.cos(fov_rad / 2.0))
            self._cos_half_fov[fov_rad] = val
        return val

    def active_models(self) -> list[str]:
        return (
            (["metric"] if self.run_metric else [])
            + (["topologic"] if self.run_topologic else [])
        )

    def print_summary(self):
        print(f"\n{'─' * 60}")
        print("Global Parameters")
        print(f"  L={self.L}, v={self.v}, dt={self.dt}")
        if self.run_metric:
            print(f"  Model 1 (Metric):      r={self.r_metric:.2f}")
        if self.run_topologic:
            print(f"  Model 2 (Topologic):   k={self.k_topologic}")
        if len(self.density_values) > 0:
            print(
                f"  Density (ρ): {np.round(self.density_values, 4)}"
                f"  →  N={[int(n) for n in self.n_values]}"
            )
        elif self.density > 0.0:
            print(f"  Density (ρ): {self.density}  →  N={self.n_values[0]}")
        print(
            f"  max_steps={self.max_steps}, noise_chunk={self.noise_chunk}, "
            f"Trials={self.num_trials}, Workers={self.num_workers}"
        )
        print(
            f"  Convergence: window={self.conv_window}, "
            f"thr={self.conv_threshold}, start={self.conv_min_steps}, "
            f"extra={self.conv_extra_steps}"
        )
        print(f"  FOV (deg): {self.fov_angles_deg}")
        print(f"  N values:  {self.n_values}")
        print(f"  η values:  {np.round(self.eta_values, 3)}  (0=무노이즈, 1=최대)")
        if len(self.v_values) > 1:
            print(f"  v values:  {np.round(self.v_values, 4)}")
        print(f"  Output:    {self.base_out_dir}")
        print(f"{'─' * 60}\n")