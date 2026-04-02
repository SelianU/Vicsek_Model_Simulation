"""
Vicsek Model — 데이터 저장 (DataWriter)

변경사항 (리뷰 반영):
  - 컬럼명: Model1→Metric, Model2→Topologic (#10, #11)
  - 파일명: {fov}_{L}_{rho}_{eta}_{v}.csv (#9)
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .config import SimConfig

# 모델명 → CSV 열 접두사 (#10, #11)
MODEL_PREFIX = {"metric": "Metric", "topologic": "Topologic"}


class DataWriter:
    """시뮬레이션 결과를 CSV로 관리하는 클래스."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self._done_cache: dict[str, set] = {}

    def csv_path(
        self, fov_deg: int, N: int, eta: float, v: float,
    ) -> str:
        """출력 경로: {fov}_{L}_{rho}_{eta}_{v}.csv (#9)"""
        L = self.cfg.L
        rho = N / (L ** 2)
        return os.path.join(
            self.cfg.base_out_dir,
            f"{fov_deg}_{L:.1f}_{rho:.4f}_{eta:.4f}_{v:.4f}.csv",
        )

    def _load_done_cache(self, csv_path: str) -> set:
        if csv_path in self._done_cache:
            return self._done_cache[csv_path]
        if not os.path.exists(csv_path):
            self._done_cache[csv_path] = set()
        else:
            try:
                cols = set(pd.read_csv(csv_path, nrows=0).columns.tolist())
            except Exception:
                cols = set()
            self._done_cache[csv_path] = cols
        return self._done_cache[csv_path]

    def already_done(self, csv_path: str, model: str, trial_idx: int) -> bool:
        col = f"{MODEL_PREFIX[model]}_Trial_{trial_idx + 1}"
        return col in self._load_done_cache(csv_path)

    def save_trial(
        self,
        csv_path: str,
        model: str,
        trial_idx: int,
        meas: np.ndarray,
    ) -> None:
        n = len(meas)
        if n == 0:
            return

        col = f"{MODEL_PREFIX[model]}_Trial_{trial_idx + 1}"
        new_col = pd.DataFrame({col: meas})

        if not os.path.exists(csv_path):
            df = pd.concat(
                [pd.DataFrame({"Time_Step": np.arange(n)}), new_col], axis=1
            )
        else:
            df = pd.read_csv(csv_path)
            if len(df) != n:
                df = pd.concat(
                    [pd.DataFrame({"Time_Step": np.arange(n)}), new_col],
                    axis=1,
                )
            else:
                df = pd.concat([df, new_col], axis=1)

        cols = ["Time_Step"] + [c for c in df.columns if c != "Time_Step"]
        df[cols].to_csv(csv_path, index=False, float_format="%.6f")
        self._load_done_cache(csv_path).add(col)