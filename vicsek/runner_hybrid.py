"""
Vicsek Model — 하이브리드 시뮬레이션 실행기 (성능 최적화 3)

Per-N 동적 라우팅: 각 시뮬레이션 조건(N)마다 최적 환경을 선택한다.
  - N < HYBRID_THRESHOLD: CPU + multiprocessing (커널 호출 오버헤드 회피)
  - N ≥ HYBRID_THRESHOLD: GPU (대규모 병렬 처리)
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time

import numpy as np

from .config import SimConfig
from .data_writer import DataWriter

# Per-N 임계값 (이 값 미만이면 CPU, 이상이면 GPU)
HYBRID_THRESHOLD = 1000


def _cpu_trial_worker(args: tuple) -> tuple:
    """CPU worker (runner_cpu.py와 동일 로직)."""
    cfg, N, fov_rad, eta, v, trial_idx, models = args
    os.environ["NUMBA_NUM_THREADS"] = "1"
    np.random.seed((os.getpid() * 2053 + trial_idx) & 0xFFFFFFFF)
    cfg.set_velocity(v)

    from .simulator_cpu import VicsekSimulatorCPU
    from .trial_runner import TrialRunner

    sim = VicsekSimulatorCPU(cfg, silent=True)
    runner = TrialRunner(sim, xp=np)
    meas = runner.run(N, fov_rad, eta, models=models)
    return trial_idx, meas


class SimulationRunnerHybrid:
    """Per-N 동적 하이브리드 라우팅으로 CPU/GPU를 유동 전환.

    N < HYBRID_THRESHOLD → CPU + mp.Pool
    N ≥ HYBRID_THRESHOLD → GPU 순차
    """

    def __init__(self, cfg: SimConfig, threshold: int = HYBRID_THRESHOLD):
        self.cfg = cfg
        self.threshold = threshold
        self.writer = DataWriter(cfg)

        # GPU 리소스 지연 초기화 (첫 GPU 구간 진입 시 생성)
        self._gpu_sim = None
        self._gpu_runner = None

    def _ensure_gpu(self):
        """GPU 시뮬레이터를 필요할 때만 초기화."""
        if self._gpu_sim is None:
            import cupy as cp
            from .simulator_gpu import VicsekSimulatorGPU
            from .trial_runner import TrialRunner

            self._gpu_sim = VicsekSimulatorGPU(self.cfg)
            self._gpu_runner = TrialRunner(self._gpu_sim, xp=cp)

    def run(self) -> None:
        cfg = self.cfg
        cfg.print_summary()
        os.makedirs(cfg.base_out_dir, exist_ok=True)

        models_active = cfg.active_models()
        total = (
            len(cfg.fov_angles_deg) * len(cfg.n_values)
            * len(cfg.eta_values) * len(cfg.v_values)
        )
        counter = 0
        t0 = time.perf_counter()

        print(f"  하이브리드 모드: N < {self.threshold} → CPU ({cfg.num_workers} workers)"
              f"  |  N ≥ {self.threshold} → GPU")
        print("=" * 26 + " STARTING DATA GENERATION " + "=" * 26)

        # CPU Pool을 전체 스캔 동안 유지 (반복 생성 비용 절감)
        pool = mp.Pool(processes=cfg.num_workers)

        try:
            for fov_idx, fov_deg in enumerate(cfg.fov_angles_deg):
                fov_rad = float(cfg._fov_rad[fov_idx])
                print(f"\n>>>> FOV: {fov_deg}° <<<<")

                for N in cfg.n_values:
                    use_gpu = (int(N) >= self.threshold)
                    env_tag = "GPU" if use_gpu else "CPU"

                    for eta_val in cfg.eta_values:
                        for v_val in cfg.v_values:
                            counter += 1
                            eta = float(eta_val)
                            v = float(v_val)
                            t_start = time.perf_counter()
                            csv = self.writer.csv_path(fov_deg, N, eta, v)

                            work_items = []
                            skipped = 0
                            for trial in range(cfg.num_trials):
                                pending = [
                                    m for m in models_active
                                    if not self.writer.already_done(csv, m, trial)
                                ]
                                if not pending:
                                    skipped += 1
                                else:
                                    work_items.append((trial, pending))

                            if not work_items:
                                print(
                                    f"  Sim {counter}/{total} [{env_tag}] "
                                    f"(FOV={fov_deg}, N={N}, η={eta:.3f}, v={v:.4f}) "
                                    f"— 전체 완료, 스킵"
                                )
                                continue

                            print(
                                f"  Sim {counter}/{total} [{env_tag}] "
                                f"(FOV={fov_deg}, N={N}, η={eta:.3f}, v={v:.4f}) "
                                f"| {len(work_items)} trials"
                                + (f" [{skipped} skipped]" if skipped else ""),
                                flush=True,
                            )

                            if use_gpu:
                                self._run_gpu_trials(
                                    cfg, N, fov_rad, eta, v,
                                    work_items, csv, models_active,
                                )
                            else:
                                self._run_cpu_trials(
                                    pool, cfg, N, fov_rad, eta, v,
                                    work_items, csv,
                                )

                            elapsed = time.perf_counter() - t_start
                            print(
                                f"\r  →  done in {elapsed:.2f}s"
                                + (f"  (skipped {skipped})" if skipped else "")
                            )
        finally:
            pool.terminate()
            pool.join()

        print("\n" + "=" * 26 + " DATA GENERATION COMPLETE " + "=" * 26)
        print(f"Total time: {(time.perf_counter() - t0) / 60:.2f} minutes.")

    # ------------------------------------------------------------------
    # CPU 경로: mp.Pool 병렬
    # ------------------------------------------------------------------

    def _run_cpu_trials(
        self, pool, cfg, N, fov_rad, eta, v, work_items, csv,
    ):
        pending_map = dict(work_items)
        task_args = [
            (cfg, N, fov_rad, eta, v, ti, pm)
            for ti, pm in work_items
        ]
        done = 0
        cs = max(1, len(work_items) // (cfg.num_workers * 4))
        for ti, meas in pool.imap_unordered(
            _cpu_trial_worker, task_args, chunksize=cs
        ):
            for m in pending_map[ti]:
                self.writer.save_trial(csv, m, ti, meas[m])
            done += 1
            sys.stdout.write(
                f"\r    진행: {done}/{len(work_items)} trials 완료"
            )
            sys.stdout.flush()

    # ------------------------------------------------------------------
    # GPU 경로: 순차 실행
    # ------------------------------------------------------------------

    def _run_gpu_trials(
        self, cfg, N, fov_rad, eta, v, work_items, csv, models_active,
    ):
        self._ensure_gpu()
        cfg.set_velocity(v)

        for trial_idx, pending in work_items:
            results = self._gpu_runner.run(N, fov_rad, eta)
            for m in pending:
                self.writer.save_trial(csv, m, trial_idx, results[m])
            sys.stdout.write(
                f"\r    GPU trial {trial_idx + 1} done"
            )
            sys.stdout.flush()