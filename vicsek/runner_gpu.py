"""
Vicsek Model — GPU 시뮬레이션 실행기
"""
from __future__ import annotations

import os
import sys
import time

import cupy as cp

from .config import SimConfig
from .data_writer import DataWriter
from .simulator_gpu import VicsekSimulatorGPU
from .trial_runner import TrialRunner


class SimulationRunnerGPU:
    """전체 파라미터 스캔을 GPU에서 실행."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.sim = VicsekSimulatorGPU(cfg)
        self.runner = TrialRunner(self.sim, xp=cp)
        self.writer = DataWriter(cfg)

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

        print("=" * 26 + " STARTING DATA GENERATION " + "=" * 26)

        for fov_idx, fov_deg in enumerate(cfg.fov_angles_deg):
            fov_rad = float(cfg._fov_rad[fov_idx])
            print(f"\n>>>> FOV: {fov_deg}° <<<<")

            for N in cfg.n_values:
                for eta_val in cfg.eta_values:
                    for v_val in cfg.v_values:
                        counter += 1
                        eta = float(eta_val)
                        v = float(v_val)
                        cfg.set_velocity(v)
                        t_start = time.perf_counter()
                        csv = self.writer.csv_path(fov_deg, N, eta, v)

                        skipped = 0
                        work_items = []
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
                                f"  Sim {counter}/{total} "
                                f"(FOV={fov_deg}, N={N}, η={eta:.3f}, v={v:.4f}) "
                                f"— 전체 완료, 스킵"
                            )
                            continue

                        print(
                            f"  Sim {counter}/{total} "
                            f"(FOV={fov_deg}, N={N}, η={eta:.3f}, v={v:.4f}) "
                            f"| {len(work_items)} trials"
                            + (f" [{skipped} skipped]" if skipped else ""),
                            flush=True,
                        )

                        job_results = {}
                        for trial, pending in work_items:
                            trial_meas = self.runner.run(N, fov_rad, eta)
                            for m in pending:
                                job_results[(m, trial)] = trial_meas[m]
                            del trial_meas
                            sys.stdout.write(
                                f"\r    진행: {len(job_results)}/{len(work_items)} trials"
                            )
                            sys.stdout.flush()

                        self.writer.save_job(csv, job_results)
                        del job_results

                        # GPU 메모리 풀 해제
                        cp.get_default_memory_pool().free_all_blocks()
                        cp.get_default_pinned_memory_pool().free_all_blocks()
                        self.writer.clear_cache(csv)

                        elapsed = time.perf_counter() - t_start
                        print(
                            f"\r  →  done in {elapsed:.2f}s"
                            + (f"  (skipped {skipped} trials)" if skipped else "")
                        )

        print("\n" + "=" * 26 + " DATA GENERATION COMPLETE " + "=" * 26)
        print(f"Total time: {(time.perf_counter() - t0) / 60:.2f} minutes.")