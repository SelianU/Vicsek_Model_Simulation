"""
Vicsek Model — CPU 시뮬레이션 실행기

변경사항: numba 스레드 제어 (#4), v 스캔 (#12), 통합 출력 (#6)
"""
from __future__ import annotations

import multiprocessing as mp
import gc
import os
import sys
import time

import numpy as np

from .config import SimConfig
from .data_writer import DataWriter
from .simulator_cpu import VicsekSimulatorCPU
from .trial_runner import TrialRunner


def _trial_worker(args: tuple) -> tuple:
    """단일 trial worker. numba prange 충돌 방지를 위해
    worker 내 NUMBA_NUM_THREADS=1 설정 (#4)."""
    cfg, N, fov_rad, eta, v, trial_idx, models = args

    # (#4) mp.Pool worker 안에서는 numba 내부 병렬을 비활성화
    os.environ["NUMBA_NUM_THREADS"] = "1"

    np.random.seed((os.getpid() * 2053 + trial_idx) & 0xFFFFFFFF)

    # 속도 적용 (#12)
    cfg.set_velocity(v)

    sim = VicsekSimulatorCPU(cfg, silent=True)
    runner = TrialRunner(sim, xp=np)
    meas = runner.run(N, fov_rad, eta, models=models)
    return trial_idx, meas


class SimulationRunnerCPU:
    """전체 파라미터 스캔을 CPU 병렬로 실행."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
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

        print(f"  병렬 workers: {cfg.num_workers}")
        print("=" * 26 + " STARTING DATA GENERATION " + "=" * 26)

        with mp.get_context("forkserver").Pool(
            processes=cfg.num_workers, maxtasksperchild=10,
        ) as pool:
            TRIAL_TIMEOUT = max(600, cfg.max_steps * 0.5)
            BATCH_SIZE = cfg.num_workers * 2  # 동시 제출 제한

            for fov_idx, fov_deg in enumerate(cfg.fov_angles_deg):
                fov_rad = float(cfg._fov_rad[fov_idx])
                print(f"\n>>>> FOV: {fov_deg}° <<<<")

                for N in cfg.n_values:
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

                            pending_map = dict(work_items)
                            job_results = {}
                            done = 0
                            failed = 0

                            # 배치 단위로 제출 → 수집 → 다음 배치
                            for b_start in range(0, len(work_items), BATCH_SIZE):
                                batch = work_items[b_start : b_start + BATCH_SIZE]

                                futures = {}
                                for ti, pm in batch:
                                    a = (cfg, N, fov_rad, eta, v, ti, pm)
                                    futures[ti] = pool.apply_async(
                                        _trial_worker, (a,)
                                    )

                                for ti, fut in futures.items():
                                    try:
                                        result_ti, meas = fut.get(
                                            timeout=TRIAL_TIMEOUT
                                        )
                                        for m in pending_map[result_ti]:
                                            job_results[(m, result_ti)] = meas[m]
                                        del meas
                                        done += 1
                                    except Exception:
                                        failed += 1
                                        done += 1
                                    sys.stdout.write(
                                        f"\r    진행: {done}/{len(work_items)} trials 완료"
                                    )
                                    sys.stdout.flush()

                                del futures
                                gc.collect()

                            # 일괄 저장
                            self.writer.save_job(csv, job_results)
                            del job_results
                            self.writer.clear_cache(csv)
                            gc.collect()

                            elapsed = time.perf_counter() - t_start
                            fail_str = f"  ({failed} failed)" if failed else ""
                            print(
                                f"\r  →  done in {elapsed:.2f}s"
                                + (f"  (skipped {skipped})" if skipped else "")
                                + fail_str
                            )

        print("\n" + "=" * 26 + " DATA GENERATION COMPLETE " + "=" * 26)
        print(f"Total time: {(time.perf_counter() - t0) / 60:.2f} minutes.")