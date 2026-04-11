"""
Vicsek Model — 하이브리드 시뮬레이션 실행기 (CPU + GPU 동시 실행)

전체 작업을 CPU 배치 / GPU 배치로 분류한 뒤 **동시에** 실행한다.
  - CPU 배치: 백그라운드 스레드에서 mp.Pool(spawn) 실행
  - GPU 배치: 메인 스레드에서 순차 실행
  - CSV 파일은 (fov, N, eta, v)마다 별개이므로 동시 쓰기 충돌 없음
"""
from __future__ import annotations

import multiprocessing as mp
import gc
import os
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

from .config import SimConfig
from .data_writer import DataWriter

HYBRID_THRESHOLD = 1000


# =========================================================================
# 작업 단위
# =========================================================================

@dataclass
class SimJob:
    """단일 시뮬레이션 조건."""
    fov_deg: int
    fov_rad: float
    N: int
    eta: float
    v: float
    csv_path: str
    work_items: list       # [(trial_idx, [pending_models]), ...]
    skipped: int


# =========================================================================
# CPU worker (spawn 안전 최상위 함수)
# =========================================================================

def _cpu_trial_worker(args: tuple) -> tuple:
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


# =========================================================================
# 스레드 세이프 로거
# =========================================================================

class _Logger:
    """CPU/GPU 스레드가 공유하는 출력 관리자.

    완료된 job은 개별 라인으로 출력하고,
    하단에 전체 진행률을 실시간 갱신한다.

    출력 예시:
      [GPU]  FOV=60  N=1600 η=0.000 v=0.5000 | 100 trials         8.5s
      [CPU]  FOV=60  N=160  η=0.000 v=0.5000 | 100 trials        12.3s
      [CPU]  FOV=60  N=160  η=0.100 v=0.5000 |  98 trials [2 skip] 11.8s
        ── CPU [2/136 jobs, 200/13600 trials]  GPU [1/220 jobs, 100/22000 trials] ──
    """

    CLEAR = "\r" + " " * 100 + "\r"   # 하단 상태줄 지우기

    def __init__(self):
        self._lock = threading.Lock()
        self._cpu_trials = 0
        self._cpu_trials_total = 0
        self._cpu_jobs = 0
        self._cpu_jobs_total = 0
        self._gpu_trials = 0
        self._gpu_trials_total = 0
        self._gpu_jobs = 0
        self._gpu_jobs_total = 0

    def set_totals(self, cpu_trials, gpu_trials, cpu_jobs, gpu_jobs):
        self._cpu_trials_total = cpu_trials
        self._gpu_trials_total = gpu_trials
        self._cpu_jobs_total = cpu_jobs
        self._gpu_jobs_total = gpu_jobs

    def tick_cpu_trial(self):
        with self._lock:
            self._cpu_trials += 1
            self._status()

    def tick_gpu_trial(self):
        with self._lock:
            self._gpu_trials += 1
            self._status()

    def job_done(self, tag: str, job: SimJob, elapsed: float):
        """job 완료 시 상세 정보를 한 줄로 출력."""
        with self._lock:
            if tag == "CPU":
                self._cpu_jobs += 1
            else:
                self._gpu_jobs += 1

            n_trials = len(job.work_items)
            skip_str = f" [{job.skipped} skip]" if job.skipped else ""

            # 하단 상태줄 지우고 → 완료 라인 출력 → 상태줄 다시 표시
            sys.stdout.write(self.CLEAR)
            sys.stdout.write(
                f"  [{tag}]  FOV={job.fov_deg:<3d} N={job.N:<5d} "
                f"η={job.eta:.3f} v={job.v:.4f} "
                f"| {n_trials:>3d} trials{skip_str:<10s} "
                f"{elapsed:>7.1f}s\n"
            )
            self._status()

    def _status(self):
        """하단 진행률 상태줄 갱신 (lock 내부에서 호출)."""
        line = (
            f"    ── CPU [{self._cpu_jobs}/{self._cpu_jobs_total} jobs, "
            f"{self._cpu_trials}/{self._cpu_trials_total} trials]  "
            f"GPU [{self._gpu_jobs}/{self._gpu_jobs_total} jobs, "
            f"{self._gpu_trials}/{self._gpu_trials_total} trials] ──"
        )
        sys.stdout.write("\r" + line)
        sys.stdout.flush()

    def final(self):
        with self._lock:
            self._status()
            print()

    def warn_failed(self, tag: str, job: SimJob, failed_trials: list):
        """타임아웃/에러 발생한 trial 경고."""
        with self._lock:
            sys.stdout.write(self.CLEAR)
            sys.stdout.write(
                f"  [{tag}]  FOV={job.fov_deg:<3d} N={job.N:<5d} "
                f"η={job.eta:.3f} v={job.v:.4f} "
                f"| {len(failed_trials)} trials TIMEOUT/FAIL "
                f"(trials: {failed_trials[:5]}{'...' if len(failed_trials)>5 else ''})\n"
            )
            self._status()


# =========================================================================
# 하이브리드 실행기
# =========================================================================

class SimulationRunnerHybrid:
    """CPU/GPU 동시 실행 하이브리드 러너.

    1단계: 전체 파라미터 공간 스캔 → 작업 목록 수집 & CPU/GPU 분류
    2단계: CPU 배치(백그라운드 스레드) + GPU 배치(메인 스레드) 동시 실행
    """

    def __init__(self, cfg: SimConfig, threshold: int = HYBRID_THRESHOLD):
        self.cfg = cfg
        self.threshold = threshold
        self.writer = DataWriter(cfg)

    def run(self) -> None:
        cfg = self.cfg
        cfg.print_summary()
        os.makedirs(cfg.base_out_dir, exist_ok=True)

        models_active = cfg.active_models()
        t0 = time.perf_counter()

        # ── 1단계: 작업 수집 & 분류 ───────────────────────────────
        cpu_jobs: list[SimJob] = []
        gpu_jobs: list[SimJob] = []
        total_skipped = 0

        for fov_idx, fov_deg in enumerate(cfg.fov_angles_deg):
            fov_rad = float(cfg._fov_rad[fov_idx])
            for N in cfg.n_values:
                for eta_val in cfg.eta_values:
                    for v_val in cfg.v_values:
                        eta = float(eta_val)
                        v = float(v_val)
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
                            total_skipped += 1
                            continue

                        job = SimJob(
                            fov_deg=fov_deg, fov_rad=fov_rad,
                            N=int(N), eta=eta, v=v,
                            csv_path=csv, work_items=work_items,
                            skipped=skipped,
                        )
                        if int(N) >= self.threshold:
                            gpu_jobs.append(job)
                        else:
                            cpu_jobs.append(job)

        cpu_trials = sum(len(j.work_items) for j in cpu_jobs)
        gpu_trials = sum(len(j.work_items) for j in gpu_jobs)

        # GPU와 동시 실행 시 CPU worker 수를 줄여 메모리 여유 확보
        cpu_workers = cfg.num_workers
        if gpu_jobs and cpu_jobs:
            cpu_workers = max(1, cfg.num_workers - 2)

        print(f"\n  동시 실행 하이브리드 모드 (임계값 N={self.threshold})")
        print(f"  ┌─ CPU: {len(cpu_jobs)} jobs, "
              f"{cpu_trials} trials ({cpu_workers} workers)")
        print(f"  └─ GPU: {len(gpu_jobs)} jobs, {gpu_trials} trials")
        if total_skipped:
            print(f"  (이미 완료: {total_skipped} jobs 스킵)")
        print("=" * 26 + " STARTING DATA GENERATION " + "=" * 26)

        if not cpu_jobs and not gpu_jobs:
            print("  모든 시뮬레이션이 이미 완료되었습니다.")
            return

        # ── 2단계: 동시 실행 ──────────────────────────────────────
        logger = _Logger()
        logger.set_totals(cpu_trials, gpu_trials,
                          len(cpu_jobs), len(gpu_jobs))

        cpu_error: list[Exception] = []

        cpu_thread = None
        if cpu_jobs:
            cpu_thread = threading.Thread(
                target=self._run_cpu_batch,
                args=(cpu_jobs, cpu_workers, logger, cpu_error),
                daemon=True,
            )
            cpu_thread.start()

        if gpu_jobs:
            self._run_gpu_batch(gpu_jobs, logger)

        if cpu_thread is not None:
            cpu_thread.join()

        logger.final()

        if cpu_error:
            print(f"\n[WARNING] CPU 배치 에러: {cpu_error[0]}")

        elapsed = time.perf_counter() - t0
        print("=" * 26 + " DATA GENERATION COMPLETE " + "=" * 26)
        print(f"Total time: {elapsed / 60:.2f} minutes.")

    # ------------------------------------------------------------------
    # CPU 배치 (백그라운드 스레드)
    # ------------------------------------------------------------------

    def _run_cpu_batch(self, jobs, cpu_workers, logger, errors):
        """배치 단위 제출 + 개별 타임아웃으로 메모리/행 방지."""
        cfg = self.cfg
        ctx = mp.get_context("forkserver")
        pool = ctx.Pool(processes=cpu_workers, maxtasksperchild=10)

        TRIAL_TIMEOUT = max(600, cfg.max_steps * 0.5)
        BATCH_SIZE = cpu_workers * 2

        try:
            for job in jobs:
                t_start = time.perf_counter()
                pending_map = dict(job.work_items)

                job_results = {}
                failed = []

                # 배치 단위로 제출 → 수집 → 다음 배치
                for b_start in range(0, len(job.work_items), BATCH_SIZE):
                    batch = job.work_items[b_start : b_start + BATCH_SIZE]

                    futures = {}
                    for ti, pm in batch:
                        args = (cfg, job.N, job.fov_rad, job.eta, job.v, ti, pm)
                        futures[ti] = pool.apply_async(
                            _cpu_trial_worker, (args,)
                        )

                    for ti, fut in futures.items():
                        try:
                            result_ti, meas = fut.get(timeout=TRIAL_TIMEOUT)
                            for m in pending_map[result_ti]:
                                job_results[(m, result_ti)] = meas[m]
                            del meas
                            logger.tick_cpu_trial()
                        except mp.TimeoutError:
                            failed.append(ti)
                            logger.tick_cpu_trial()
                        except Exception:
                            failed.append(ti)
                            logger.tick_cpu_trial()

                    del futures
                    gc.collect()

                # 일괄 저장
                self.writer.save_job(job.csv_path, job_results)
                del job_results
                self.writer.clear_cache(job.csv_path)
                gc.collect()

                elapsed = time.perf_counter() - t_start

                if failed:
                    job.skipped += len(failed)
                    logger.warn_failed("CPU", job, failed)

                logger.job_done("CPU", job, elapsed)

        except Exception as e:
            errors.append(e)
        finally:
            pool.terminate()
            pool.join()

    # ------------------------------------------------------------------
    # GPU 배치 (메인 스레드)
    # ------------------------------------------------------------------

    def _run_gpu_batch(self, jobs, logger):
        import cupy as cp
        from .simulator_gpu import VicsekSimulatorGPU
        from .trial_runner import TrialRunner

        cfg = self.cfg
        sim = VicsekSimulatorGPU(cfg)
        runner = TrialRunner(sim, xp=cp)
        mempool = cp.get_default_memory_pool()
        pinned_pool = cp.get_default_pinned_memory_pool()

        for job in jobs:
            t_start = time.perf_counter()
            cfg.set_velocity(job.v)

            # job 단위로 결과 수집 후 일괄 저장
            job_results = {}  # {(model, trial_idx): meas_array}

            for trial_idx, pending in job.work_items:
                trial_meas = runner.run(job.N, job.fov_rad, job.eta)
                for m in pending:
                    job_results[(m, trial_idx)] = trial_meas[m]
                del trial_meas
                logger.tick_gpu_trial()

            # 일괄 저장 (CSV read/write 1회)
            self.writer.save_job(job.csv_path, job_results)
            del job_results

            # GPU 메모리 풀 해제 (OS에 반환)
            mempool.free_all_blocks()
            pinned_pool.free_all_blocks()

            # done_cache 해당 키 정리
            self.writer.clear_cache(job.csv_path)

            elapsed = time.perf_counter() - t_start
            logger.job_done("GPU", job, elapsed)