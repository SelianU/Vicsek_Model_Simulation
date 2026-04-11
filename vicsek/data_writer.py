"""
Vicsek Model — 데이터 저장 (DataWriter)

메모리 최적화:
  - save_job(): job 단위 일괄 저장 (CSV read/write 1회로 통합)
  - save_trial(): 개별 저장 (호환성 유지, 매번 명시적 del)
  - _done_cache를 job 완료 후 해당 키만 유지
"""
from __future__ import annotations

import gc
import os

import numpy as np
import pandas as pd

from .config import SimConfig

MODEL_PREFIX = {"metric": "Metric", "topologic": "Topologic"}


class DataWriter:
    """시뮬레이션 결과를 CSV로 관리하는 클래스."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self._done_cache: dict[str, set] = {}

    def csv_path(
        self, fov_deg: int, N: int, eta: float, v: float,
    ) -> str:
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

    def clear_cache(self, csv_path: str | None = None) -> None:
        """메모리 절약을 위해 완료된 캐시를 정리한다."""
        if csv_path is None:
            self._done_cache.clear()
        else:
            self._done_cache.pop(csv_path, None)

    # ------------------------------------------------------------------
    # 일괄 저장 (권장): job 단위로 모든 trial을 한 번에 쓴다
    # ------------------------------------------------------------------

    def save_job(
        self,
        csv_path: str,
        results: dict[tuple[str, int], np.ndarray],
    ) -> None:
        """한 job의 전체 결과를 CSV에 일괄 저장한다.

        Args:
            csv_path: CSV 파일 경로
            results: {(model, trial_idx): meas_array, ...}
                     meas_array.shape = (conv_extra_steps,)
        """
        if not results:
            return

        # 새 컬럼 구성
        new_cols = {}
        n = None
        for (model, trial_idx), meas in results.items():
            if len(meas) == 0:
                continue
            if n is None:
                n = len(meas)
            col = f"{MODEL_PREFIX[model]}_Trial_{trial_idx + 1}"
            new_cols[col] = meas

        if not new_cols or n is None:
            return

        # 기존 CSV 로드 (있으면)
        if os.path.exists(csv_path):
            try:
                df = pd.read_csv(csv_path)
                if len(df) != n:
                    # 행 수 불일치 → 리셋
                    df = pd.DataFrame({"Time_Step": np.arange(n)})
            except Exception:
                df = pd.DataFrame({"Time_Step": np.arange(n)})
        else:
            df = pd.DataFrame({"Time_Step": np.arange(n)})

        # 새 컬럼을 한 번에 합치기 (fragmentation 방지)
        new_df = pd.DataFrame(new_cols, index=df.index)
        df = pd.concat([df, new_df], axis=1)

        # 정렬 & 저장
        cols = ["Time_Step"] + sorted(
            [c for c in df.columns if c != "Time_Step"]
        )
        df[cols].to_csv(csv_path, index=False, float_format="%.6f")

        # 캐시 갱신
        cache = self._load_done_cache(csv_path)
        cache.update(new_cols.keys())

        # 명시적 메모리 해제
        del df
        gc.collect()

    # ------------------------------------------------------------------
    # 개별 저장 (호환용): 한 trial씩 저장
    # ------------------------------------------------------------------

    def save_trial(
        self,
        csv_path: str,
        model: str,
        trial_idx: int,
        meas: np.ndarray,
    ) -> None:
        """단일 trial 저장 (기존 호환). save_job()보다 느림."""
        n = len(meas)
        if n == 0:
            return

        col = f"{MODEL_PREFIX[model]}_Trial_{trial_idx + 1}"
        new_col = pd.DataFrame({col: meas})

        if not os.path.exists(csv_path):
            df = pd.concat(
                [pd.DataFrame({"Time_Step": np.arange(n)}), new_col],
                axis=1,
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

        # 명시적 메모리 해제
        del df, new_col