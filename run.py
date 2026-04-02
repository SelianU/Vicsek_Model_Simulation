#!/usr/bin/env python3
"""
Vicsek Model — 통합 진입점

변경사항 (리뷰 반영):
  - 단일 진입점 통합 (#7): N 기반 CPU/GPU 자동 라우팅
  - --visualize는 GPU 전용 (#8)
  - --force-cpu / --force-gpu 로 수동 오버라이드

사용 예시:
  # 자동 라우팅 (GPU 사용 가능하면 대규모 N에서 GPU 사용)
  python run.py --topologic --eta-auto

  # CPU 강제
  python run.py --force-cpu --topologic --eta-auto

  # GPU 시각화
  python run.py --visualize --viz-model topologic --viz-N 300 --viz-eta 0.5

  # 속도 스캔 (#12)
  python run.py --topologic --eta-auto --v 0.05 0.1 0.2 0.5
"""
from __future__ import annotations

import multiprocessing as mp
import sys

import numpy as np

from vicsek.cli import build_parser, build_cfg
from vicsek.config import SimConfig


def _gpu_available() -> bool:
    """CuPy + CUDA 사용 가능 여부 확인."""
    try:
        import cupy as cp
        cp.cuda.Device().use()
        return True
    except Exception:
        return False


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # ── 시각화 모드: GPU 전용 (#8) ─────────────────────────────
    if args.visualize:
        if not _gpu_available():
            sys.exit(
                "[ERROR] --visualize는 GPU 전용입니다. "
                "CuPy와 CUDA가 필요합니다."
            )

        _d = SimConfig()
        viz_L = args.viz_L if args.viz_L is not None else _d.L

        if args.viz_density is not None:
            viz_N = max(2, round(args.viz_density * viz_L ** 2))
            print(f"  ρ={args.viz_density}  L={viz_L}  →  N={viz_N}")
        else:
            viz_N = args.viz_N if args.viz_N is not None else 200

        cfg = SimConfig(
            run_metric=(args.viz_model == "metric"),
            run_topologic=(args.viz_model == "topologic"),
            L=viz_L,
        )

        fov_rad = float(np.deg2rad(args.viz_fov))
        if fov_rad not in cfg._cos_half_fov:
            cfg._cos_half_fov[fov_rad] = float(np.cos(fov_rad / 2.0))

        model_label = "METRIC" if args.viz_model == "metric" else "TOPOLOGIC"
        print(
            f"\n[Visualization Mode]  model={model_label}"
            f"  N={viz_N}  FOV={args.viz_fov}°  η={args.viz_eta}"
            f"  L={viz_L}"
        )

        from vicsek.visualization import VisualizationRunner
        VisualizationRunner(cfg).run(
            N=viz_N, fov_deg=args.viz_fov, eta=args.viz_eta,
            model=args.viz_model, max_frames=args.viz_frames,
            interval_ms=args.viz_interval, save_path=args.viz_save,
        )
        return

    # ── 데이터 수집 모드 ───────────────────────────────────────
    cfg = build_cfg(args)

    # 실행 환경 결정
    if args.force_gpu:
        if not _gpu_available():
            sys.exit("[ERROR] --force-gpu 지정했으나 GPU를 사용할 수 없습니다.")
        from vicsek.runner_gpu import SimulationRunnerGPU
        SimulationRunnerGPU(cfg).run()

    elif args.force_cpu:
        from vicsek.runner_cpu import SimulationRunnerCPU
        SimulationRunnerCPU(cfg).run()

    else:
        # 자동 라우팅: GPU 사용 가능하면 Per-N 하이브리드 (#최적화3)
        if _gpu_available():
            from vicsek.runner_hybrid import SimulationRunnerHybrid, HYBRID_THRESHOLD
            min_N, max_N = int(cfg.n_values.min()), int(cfg.n_values.max())
            if min_N >= HYBRID_THRESHOLD:
                # 모든 N이 임계값 이상 → GPU 전용
                print(f"[AUTO] 모든 N ≥ {HYBRID_THRESHOLD}: GPU 전용 모드")
                from vicsek.runner_gpu import SimulationRunnerGPU
                SimulationRunnerGPU(cfg).run()
            elif max_N < HYBRID_THRESHOLD:
                # 모든 N이 임계값 미만 → CPU 전용
                print(f"[AUTO] 모든 N < {HYBRID_THRESHOLD}: CPU 전용 모드")
                from vicsek.runner_cpu import SimulationRunnerCPU
                SimulationRunnerCPU(cfg).run()
            else:
                # N 범위가 임계값을 걸침 → Per-N 하이브리드
                print(
                    f"[AUTO] N 범위 [{min_N}..{max_N}]: "
                    f"하이브리드 모드 (임계값={HYBRID_THRESHOLD})"
                )
                SimulationRunnerHybrid(cfg).run()
        else:
            print("[AUTO] GPU 미감지: CPU 전용 모드")
            from vicsek.runner_cpu import SimulationRunnerCPU
            SimulationRunnerCPU(cfg).run()


if __name__ == "__main__":
    mp.freeze_support()
    main()