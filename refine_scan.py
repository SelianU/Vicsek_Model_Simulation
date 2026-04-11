#!/usr/bin/env python3
"""
refine_scan.py — η_c 주변 정밀 스캔

extract_eta_c.py → eta_c_summary.csv → 이 스크립트 → run.py

각 (model, FOV, ρ)마다 η_c가 다르므로 **개별 시뮬레이션**을 돌린다.
  - metric  FOV=180 ρ=0.5 η_c=0.38 → η ∈ [0.28, 0.48] 로 metric만 실행
  - topologic FOV=180 ρ=0.5 η_c=0.42 → η ∈ [0.32, 0.52] 로 topologic만 실행

사용법:
  python refine_scan.py                         # 기본: L×2, η_c±0.1, 21점
  python refine_scan.py --dry-run               # 실행 안 하고 계획만 출력
  python refine_scan.py --L-scale 3             # L을 3배로
  python refine_scan.py --eta-width 0.05        # η_c 주변 ±0.05
  python refine_scan.py --eta-steps 31          # 구간 내 31개 포인트
  python refine_scan.py --fov 180 --rho 1.0     # 특정 조건만
  python refine_scan.py -- --workers 8          # run.py에 추가 인자 전달
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd


# =========================================================================
# 작업 단위
# =========================================================================

@dataclass
class RefineJob:
    """단일 (model, fov, rho)에 대한 정밀 스캔 작업."""
    model: str       # "metric" or "topologic"
    fov: int
    rho: float
    eta_c: float     # 원래 η_c (χ 최대 위치)
    chi_max: float
    L_old: float
    L_new: float
    v: float
    N_new: int       # ρ × L_new²
    eta_range: np.ndarray  # 세분화된 η 배열


# =========================================================================
# Summary 로드 & 작업 생성
# =========================================================================

def load_and_build_jobs(
    summary_path: str,
    L_scale: float,
    eta_width: float,
    eta_steps: int,
    fov_filter: list[int] | None,
    rho_filter: list[float] | None,
    model_filter: list[str] | None,
) -> list[RefineJob]:
    """eta_c_summary.csv를 읽고 RefineJob 리스트를 생성한다."""

    df = pd.read_csv(summary_path)

    # 필수 컬럼 확인
    required = {"model", "fov", "rho", "L", "eta_c"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(
            f"[ERROR] CSV에 필수 컬럼 없음: {missing}\n"
            f"  extract_eta_c.py를 먼저 실행하세요."
        )

    # v 컬럼 처리: 없으면 기본값 0.1
    if "v" not in df.columns:
        print("  [WARN] summary에 v 컬럼 없음 → v=0.1 가정")
        df["v"] = 0.1

    if "chi_max" not in df.columns:
        df["chi_max"] = np.nan

    # 필터 적용
    if fov_filter:
        df = df[df["fov"].isin(fov_filter)]
    if rho_filter:
        df = df[df["rho"].isin(rho_filter)]
    if model_filter:
        df = df[df["model"].isin(model_filter)]

    # eta_c가 NaN인 행 제거
    df = df.dropna(subset=["eta_c"])

    if df.empty:
        sys.exit("[ERROR] 필터 후 데이터가 없습니다.")

    # 작업 생성
    jobs = []
    for _, row in df.iterrows():
        L_old = float(row["L"])
        L_new = L_old * L_scale
        rho = float(row["rho"])
        eta_c = float(row["eta_c"])
        v = float(row["v"])

        # η_c 주변 세분화 배열
        lo = max(0.0, eta_c - eta_width)
        hi = min(1.0, eta_c + eta_width)
        eta_range = np.round(np.linspace(lo, hi, eta_steps), 6)

        jobs.append(RefineJob(
            model=str(row["model"]),
            fov=int(row["fov"]),
            rho=rho,
            eta_c=eta_c,
            chi_max=float(row["chi_max"]) if not np.isnan(row["chi_max"]) else 0.0,
            L_old=L_old,
            L_new=L_new,
            v=v,
            N_new=max(2, round(rho * L_new ** 2)),
            eta_range=eta_range,
        ))

    return jobs


# =========================================================================
# 계획 출력
# =========================================================================

def print_plan(jobs: list[RefineJob]):
    if not jobs:
        print("  작업 없음.")
        return

    L_old = jobs[0].L_old
    L_new = jobs[0].L_new

    print(f"\n{'='*70}")
    print(f"  정밀 스캔 계획  |  {len(jobs)} 개별 시뮬레이션")
    print(f"{'='*70}")
    print(f"  L: {L_old} → {L_new}  (×{L_new/L_old:.1f})")
    print()
    print(f"  {'#':>3s}  {'model':10s} {'FOV':>4s} {'ρ':>6s} {'N':>6s}"
          f"  {'η_c':>6s} {'χ_max':>8s}  {'η 범위':>20s}  {'포인트':>4s}")
    print(f"  {'─'*68}")

    total_points = 0
    for i, j in enumerate(jobs):
        total_points += len(j.eta_range)
        print(
            f"  {i+1:3d}  {j.model:10s} {j.fov:4d} {j.rho:6.3f} {j.N_new:6d}"
            f"  {j.eta_c:6.4f} {j.chi_max:8.2f}"
            f"  [{j.eta_range[0]:.4f}, {j.eta_range[-1]:.4f}]"
            f"  {len(j.eta_range):4d}"
        )

    v_set = sorted(set(j.v for j in jobs))
    print(f"\n  v: {v_set}")
    print(f"  총 (model, FOV, ρ, η) 조합: {total_points}")
    print(f"{'='*70}")


# =========================================================================
# 실행
# =========================================================================

def build_command(job: RefineJob, output_dir: str, extra_args: list[str]) -> list[str]:
    """단일 RefineJob에 대한 run.py 명령어를 구성한다."""
    model_flag = "--metric" if job.model == "metric" else "--topologic"

    cmd = [
        sys.executable, "run.py",
        model_flag,
        "--L", str(job.L_new),
        "--fov", str(job.fov),
        "--density", str(job.rho),
        "--eta", *[f"{e:.6f}" for e in job.eta_range],
        "--v", str(job.v),
        "--output-dir", output_dir,
    ]
    cmd.extend(extra_args)
    return cmd


def format_command(cmd: list[str]) -> str:
    """명령어를 보기 좋게 포맷."""
    parts = [f"{cmd[0]} {cmd[1]}"]
    i = 2
    while i < len(cmd):
        if cmd[i].startswith("--"):
            parts.append(f"  {cmd[i]}")
            # 다음이 값이면 같은 줄에
            if i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
                # --eta 뒤에 많은 값이 올 수 있음
                vals = []
                i += 1
                while i < len(cmd) and not cmd[i].startswith("--"):
                    vals.append(cmd[i])
                    i += 1
                if len(vals) <= 3:
                    parts[-1] += " " + " ".join(vals)
                else:
                    parts[-1] += f" {vals[0]} ... {vals[-1]} ({len(vals)}개)"
                continue
        i += 1
    return " \\\n    ".join(parts)


def run_jobs(
    jobs: list[RefineJob],
    output_dir: str,
    extra_args: list[str],
    dry_run: bool,
):
    """모든 작업을 순차 실행한다."""

    if dry_run:
        print(f"\n  (--dry-run: 명령어만 출력)")
        for i, job in enumerate(jobs):
            cmd = build_command(job, output_dir, extra_args)
            print(f"\n  [{i+1}/{len(jobs)}] {job.model} FOV={job.fov} ρ={job.rho}")
            print(f"  {format_command(cmd)}")
        return

    print(f"\n{'='*70}")
    print(f"  시뮬레이션 시작 ({len(jobs)} jobs)")
    print(f"{'='*70}\n")

    t0 = time.perf_counter()

    for i, job in enumerate(jobs):
        cmd = build_command(job, output_dir, extra_args)

        print(
            f"  [{i+1}/{len(jobs)}]  {job.model:10s} FOV={job.fov} "
            f"ρ={job.rho:.3f} (N={job.N_new}) "
            f"η∈[{job.eta_range[0]:.4f},{job.eta_range[-1]:.4f}] "
            f"v={job.v}",
            flush=True,
        )

        t_start = time.perf_counter()
        result = subprocess.run(cmd)
        elapsed = time.perf_counter() - t_start

        if result.returncode != 0:
            print(f"    [FAIL] exit code {result.returncode}")
        else:
            print(f"    done in {elapsed:.1f}s")

    total = time.perf_counter() - t0
    print(f"\n{'='*70}")
    print(f"  전체 완료: {total / 60:.1f} minutes")
    print(f"{'='*70}")


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="η_c 주변 정밀 스캔 (모델별 개별 실행)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python refine_scan.py                          # 기본
  python refine_scan.py --dry-run                # 계획만 출력
  python refine_scan.py --L-scale 3              # L을 3배로
  python refine_scan.py --eta-width 0.05         # η_c ± 0.05
  python refine_scan.py --fov 180 --rho 1.0      # 특정 조건만
  python refine_scan.py --model topologic        # 특정 모델만
  python refine_scan.py -- --workers 8           # run.py 추가 인자
        """,
    )
    parser.add_argument(
        "--summary", default="eta_c_summary.csv",
        help="extract_eta_c.py 출력 (기본값: eta_c_summary.csv)",
    )
    parser.add_argument(
        "--L-scale", type=float, default=2.0,
        help="L 배율 (기본값: 2.0)",
    )
    parser.add_argument(
        "--eta-width", type=float, default=0.1,
        help="η_c 주변 반폭 (기본값: 0.1)",
    )
    parser.add_argument(
        "--eta-steps", type=int, default=21,
        help="구간 내 포인트 수 (기본값: 21)",
    )
    parser.add_argument(
        "--fov", type=int, nargs="+", default=None,
        help="특정 FOV만",
    )
    parser.add_argument(
        "--rho", type=float, nargs="+", default=None,
        help="특정 ρ만",
    )
    parser.add_argument(
        "--model", type=str, nargs="+", default=None,
        choices=["metric", "topologic"],
        help="특정 모델만",
    )
    parser.add_argument(
        "--output-dir", default="Vicsek_Results_Refined",
        help="출력 디렉토리 (기본값: Vicsek_Results_Refined)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="실행하지 않고 계획만 출력",
    )
    parser.add_argument(
        "extra", nargs="*",
        help="run.py에 전달할 추가 인자 (예: --workers 8)",
    )

    args = parser.parse_args()

    print(f"Loading {args.summary}...")
    jobs = load_and_build_jobs(
        summary_path=args.summary,
        L_scale=args.L_scale,
        eta_width=args.eta_width,
        eta_steps=args.eta_steps,
        fov_filter=args.fov,
        rho_filter=args.rho,
        model_filter=args.model,
    )
    print(f"  {len(jobs)} jobs 생성.")

    print_plan(jobs)
    run_jobs(jobs, args.output_dir, args.extra or [], args.dry_run)


if __name__ == "__main__":
    main()