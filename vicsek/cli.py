"""
Vicsek Model — CLI 파싱 및 SimConfig 생성

변경사항: knn→topologic (#11), v 스캔 (#12), 통합 출력 (#6), GPU 자동 라우팅 (#7)
"""
from __future__ import annotations

import argparse
import os

import numpy as np

from .config import SimConfig


def build_parser(
    description: str = "Vicsek Model Simulation",
) -> argparse.ArgumentParser:
    """모든 인자가 등록된 통합 ArgumentParser."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _d = SimConfig()

    # ── FOV ──
    parser.add_argument(
        "--fov", type=int, nargs="+", default=None, metavar="DEG",
        help=f"FOV 각도 목록 (도, 기본값: {list(_d.fov_angles_deg)})",
    )

    # ── N 관련 ──
    n_grp = parser.add_mutually_exclusive_group()
    n_grp.add_argument("--N", type=int, nargs="+", metavar="N", help="입자 수 목록")
    n_grp.add_argument("--N-range", type=int, nargs=3, metavar=("START", "STOP", "STEP"))
    n_grp.add_argument("--density", type=float, nargs="+", metavar="RHO")
    n_grp.add_argument("--density-range", type=float, nargs=3, metavar=("START", "STOP", "STEP"))

    parser.add_argument("--L", type=float, default=None, help=f"시스템 크기 (기본값: {_d.L})")

    # ── η (0~1 정규화) ──
    eta_grp = parser.add_mutually_exclusive_group()
    eta_grp.add_argument(
        "--eta", type=float, nargs="+", metavar="ETA",
        help="노이즈 η 값 (0=무노이즈, 1=최대)",
    )
    eta_grp.add_argument(
        "--eta-auto", action="store_true",
        help="η 자동 생성: 0.0, 0.05, ..., 1.0 (21개)",
    )

    # ── v 스캔 (#12) ──
    v_grp = parser.add_mutually_exclusive_group()
    v_grp.add_argument(
        "--v", type=float, nargs="+", metavar="V",
        help="입자 속도 목록 (예: --v 0.05 0.1 0.2)",
    )
    v_grp.add_argument(
        "--v-range", type=float, nargs=3, metavar=("START", "STOP", "STEP"),
        help="입자 속도 범위 (예: --v-range 0.05 0.5 0.05)",
    )

    # ── 모델 선택 (#11) ──
    parser.add_argument("--metric", action="store_true", help="Metric 모델 실행")
    parser.add_argument("--topologic", action="store_true", help="Topologic 모델 실행")

    # ── 실행 환경 (#7) ──
    env_grp = parser.add_mutually_exclusive_group()
    env_grp.add_argument("--force-cpu", action="store_true", help="CPU 강제 사용")
    env_grp.add_argument("--force-gpu", action="store_true", help="GPU 강제 사용")

    # ── 시각화 (#8: GPU 전용) ──
    parser.add_argument("--visualize", action="store_true", help="시각화 모드 (GPU 전용)")
    parser.add_argument("--viz-model", type=str, default="topologic", choices=["metric", "topologic"])
    parser.add_argument("--viz-N", type=int, default=None)
    parser.add_argument("--viz-density", type=float, default=None, metavar="RHO")
    parser.add_argument("--viz-L", type=float, default=None)
    parser.add_argument("--viz-fov", type=int, default=360)
    parser.add_argument("--viz-eta", type=float, default=0.5, help="시각화 η (0~1, 기본값: 0.5)")
    parser.add_argument("--viz-frames", type=int, default=2000)
    parser.add_argument("--viz-interval", type=int, default=30)
    parser.add_argument("--viz-save", type=str, default=None, metavar="FILE")

    # ── 공통 옵션 ──
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)

    return parser


def build_cfg(args: argparse.Namespace) -> SimConfig:
    """CLI 인자로부터 SimConfig를 생성."""
    defaults = SimConfig()

    L = args.L if args.L is not None else defaults.L

    # N / density
    density_vals = np.array([], dtype=np.float64)
    density_single = 0.0
    n_values = defaults.n_values

    if hasattr(args, "N") and args.N:
        n_values = np.array(sorted(args.N), dtype=np.int32)
    elif hasattr(args, "N_range") and args.N_range:
        s, e, st = args.N_range
        n_values = np.arange(s, e + 1, st, dtype=np.int32)
    elif hasattr(args, "density") and args.density:
        density_vals = np.array(sorted(args.density), dtype=np.float64)
    elif hasattr(args, "density_range") and args.density_range:
        s, e, st = args.density_range
        density_vals = np.arange(s, e + st * 0.5, st, dtype=np.float64)

    # η
    if args.eta:
        eta_values = np.array(sorted(args.eta))
    elif args.eta_auto:
        eta_values = np.linspace(0.0, 1.0, 21)
    else:
        eta_values = defaults.eta_values

    # v (#12)
    if hasattr(args, "v") and args.v:
        v_values = np.array(sorted(args.v))
    elif hasattr(args, "v_range") and args.v_range:
        s, e, st = args.v_range
        v_values = np.arange(s, e + st * 0.5, st)
    else:
        v_values = defaults.v_values

    # 모델 (#11)
    run_metric = args.metric
    run_topologic = args.topologic
    if not run_metric and not run_topologic:
        run_topologic = True

    return SimConfig(
        run_metric=run_metric,
        run_topologic=run_topologic,
        L=L,
        v=float(v_values[0]),
        fov_angles_deg=(
            np.array(sorted(args.fov), dtype=np.int32)
            if args.fov else defaults.fov_angles_deg
        ),
        n_values=n_values,
        density=density_single,
        density_values=density_vals,
        eta_values=eta_values,
        v_values=v_values,
        base_out_dir=(
            args.output_dir if args.output_dir
            else defaults.base_out_dir  # 통합: Vicsek_Results (#6)
        ),
        num_trials=args.trials if args.trials else defaults.num_trials,
        max_steps=args.max_steps if args.max_steps else defaults.max_steps,
        noise_chunk=args.chunk if args.chunk else defaults.noise_chunk,
        num_workers=args.workers if args.workers else -1,
    )