"""
extract_eta_c.py

각 (model, FOV, ρ) 조건에서 χ(susceptibility)가 최대인 η를 η_c로 추출한다.

새 파일 형식 대응:
  - CSV 파일명: {fov}_{L}_{rho}_{eta}_{v}.csv
  - 컬럼명: Metric_Trial_*, Topologic_Trial_*
  - η ∈ [0, 1] 정규화

출력:
  1. eta_c_summary.csv — 전체 결과 테이블
  2. eta_c_curves.png  — χ vs η 곡선 (FOV별 패널, ρ별 라인)
  3. delta_eta_c.png   — Δη_c = η_c(Topologic) − η_c(Metric) 히트맵
  4. 터미널 요약 출력

사용법:
  python extract_eta_c.py
  python extract_eta_c.py --folder Vicsek_Results --v 0.5
"""
from __future__ import annotations

import argparse
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================================
# 데이터 로드 (order_parameter.py와 동일한 파싱)
# =========================================================================

def parse_filename(filename):
    stem = os.path.basename(filename).replace(".csv", "")
    parts = stem.split("_")
    if len(parts) != 5:
        return None
    try:
        return {
            "fov": int(parts[0]),
            "L": float(parts[1]),
            "rho": float(parts[2]),
            "eta": float(parts[3]),
            "v": float(parts[4]),
        }
    except ValueError:
        return None


def load_data(folder):
    dfs = {}
    for path in glob.glob(os.path.join(folder, "*.csv")):
        p = parse_filename(path)
        if p is None:
            continue
        try:
            df = pd.read_csv(path)
            if len(df) > 0:
                key = (p["fov"], p["L"], p["rho"], p["eta"], p["v"])
                dfs[key] = df
        except Exception:
            continue
    return dfs


# =========================================================================
# Order Parameter 계산
# =========================================================================

def calculate_order_parameter(dfs):
    values = {}
    for key, df in dfs.items():
        fov, L, rho, eta, v = key
        N = max(2, round(rho * L ** 2))
        result = {}

        for prefix, label in [("Metric", "metric"), ("Topologic", "topologic")]:
            cols = df.filter(regex=prefix)
            if cols.empty:
                continue
            phi_mean = cols.mean()
            phi2_mean = cols.pow(2).mean()
            phi4_mean = cols.pow(4).mean()

            result[f"{label}_mean"] = phi_mean.mean()
            result[f"{label}_chi"] = N * (phi2_mean.mean() - phi_mean.pow(2).mean())
            result[f"{label}_binder"] = 1 - (phi4_mean.mean() / (3 * phi2_mean.mean() ** 2))

        if result:
            values[key] = result
    return values


# =========================================================================
# η_c 추출: 각 (model, fov, ρ)에서 χ 최대값의 η
# =========================================================================

def extract_eta_c(values, v_filter):
    """각 (model, fov, rho)에서 chi가 최대인 eta를 η_c로 추출."""

    # 사용 가능한 축 감지
    keys = list(values.keys())
    fov_set = sorted(set(k[0] for k in keys))
    rho_set = sorted(set(k[2] for k in keys))
    eta_set = sorted(set(k[3] for k in keys))
    L_val = keys[0][1]

    # v 필터
    filtered = {k: v for k, v in values.items() if abs(k[4] - v_filter) < 1e-6}
    if not filtered:
        print(f"[ERROR] v={v_filter}에 해당하는 데이터 없음")
        return {}, {}

    # 모델 감지
    sample = next(iter(filtered.values()))
    models = []
    if "metric_chi" in sample:
        models.append("metric")
    if "topologic_chi" in sample:
        models.append("topologic")

    results = {}
    curves = {}  # (model, fov, rho) → (eta_arr, chi_arr)

    for model in models:
        chi_key = f"{model}_chi"
        for fov in fov_set:
            for rho in rho_set:
                eta_arr = []
                chi_arr = []

                for eta in eta_set:
                    entry = filtered.get((fov, L_val, rho, eta, v_filter))
                    if entry and chi_key in entry:
                        eta_arr.append(eta)
                        chi_arr.append(entry[chi_key])

                if not eta_arr:
                    continue

                eta_arr = np.array(eta_arr)
                chi_arr = np.array(chi_arr)
                curves[(model, fov, rho)] = (eta_arr, chi_arr)

                # χ 최대값 위치 = η_c
                idx = np.argmax(chi_arr)
                N = max(2, round(rho * L_val ** 2))

                results[(model, fov, rho)] = {
                    "eta_c": float(eta_arr[idx]),
                    "chi_max": float(chi_arr[idx]),
                    "N": N,
                    "L": L_val,
                    "v": v_filter,
                }

    return results, curves


# =========================================================================
# 터미널 출력
# =========================================================================

def print_summary(results):
    print(f"\n{'='*60}")
    print(f"  η_c 요약 (χ 최대값 기준)")
    print(f"{'='*60}")
    print(f"  {'model':10s} {'FOV':>5s} {'ρ':>6s} {'N':>5s}  {'η_c':>6s}  {'χ_max':>9s}")
    print("  " + "-" * 52)

    for (model, fov, rho), r in sorted(results.items()):
        print(
            f"  {model:10s} {fov:5d} {rho:6.3f} {r['N']:5d}  "
            f"{r['eta_c']:6.3f}  {r['chi_max']:9.3f}"
        )


# =========================================================================
# χ vs η 곡선 플롯
# =========================================================================

def plot_chi_curves(results, curves):
    fov_set = sorted(set(k[1] for k in curves.keys()))
    rho_set = sorted(set(k[2] for k in curves.keys()))
    models = sorted(set(k[0] for k in curves.keys()))

    n_rows = len(models)
    n_cols = len(fov_set)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    fig.suptitle("χ (susceptibility) vs η", fontsize=14, fontweight="bold")

    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, len(rho_set) - 1)) for i in range(len(rho_set))]

    for mi, model in enumerate(models):
        for fi, fov in enumerate(fov_set):
            ax = axes[mi, fi]

            for ri, rho in enumerate(rho_set):
                curve = curves.get((model, fov, rho))
                if curve is None:
                    continue

                eta_arr, chi_arr = curve
                r = results.get((model, fov, rho))
                N = r["N"] if r else "?"

                ax.plot(
                    eta_arr, chi_arr,
                    "o-", color=colors[ri], markersize=3, linewidth=1.2,
                    label=f"ρ={rho} (N={N})",
                )

                # η_c 마커
                if r:
                    ax.axvline(
                        r["eta_c"], color=colors[ri],
                        linewidth=0.8, linestyle="--", alpha=0.5,
                    )

            ax.set_xlabel("η", fontsize=9)
            ax.set_ylabel("χ", fontsize=9)
            ax.set_title(
                f"{model.capitalize()}  |  FOV={fov}°",
                fontsize=10, fontweight="bold",
            )
            ax.grid(True, alpha=0.3)

            if fi == n_cols - 1:
                ax.legend(fontsize=5.5, loc="upper right")

    plt.savefig("eta_c_curves.png", dpi=300, bbox_inches="tight")
    print("  저장: eta_c_curves.png")
    plt.close(fig)


# =========================================================================
# Δη_c 히트맵
# =========================================================================

def plot_delta_eta_c(results):
    # metric과 topologic 둘 다 있는지 확인
    models = set(k[0] for k in results.keys())
    if "metric" not in models or "topologic" not in models:
        print("  [SKIP] Δη_c: metric 또는 topologic 데이터 부족")
        return

    fov_set = sorted(set(k[1] for k in results.keys()))
    rho_set = sorted(set(k[2] for k in results.keys()))

    L_val = next(iter(results.values()))["L"]

    fig, axes = plt.subplots(
        1, len(fov_set),
        figsize=(3.5 * len(fov_set), 3.5),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes[0]
    fig.suptitle(
        "Δη_c = η_c(Topologic) − η_c(Metric)",
        fontsize=13, fontweight="bold",
    )

    all_deltas = []
    grids = {}
    for fov in fov_set:
        grid = np.full(len(rho_set), np.nan)
        for ri, rho in enumerate(rho_set):
            rt = results.get(("topologic", fov, rho))
            rm = results.get(("metric", fov, rho))
            if rt and rm:
                grid[ri] = rt["eta_c"] - rm["eta_c"]
        grids[fov] = grid
        all_deltas.extend(grid[~np.isnan(grid)].tolist())

    vabs = max(abs(min(all_deltas)), abs(max(all_deltas))) if all_deltas else 1.0

    for fi, fov in enumerate(fov_set):
        ax = axes[fi]
        grid = grids[fov].reshape(1, -1)
        im = ax.imshow(grid, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="auto")

        labels = [f"{r}\n(N={max(2, round(r * L_val**2))})" for r in rho_set]
        ax.set_xticks(range(len(rho_set)))
        ax.set_xticklabels(labels, fontsize=6, ha="center")
        ax.set_xlabel("ρ (N)", fontsize=8)
        ax.set_yticks([])
        ax.set_title(f"FOV={fov}°", fontsize=10, fontweight="bold")

        for ri in range(len(rho_set)):
            val = grids[fov][ri]
            if not np.isnan(val):
                color = "white" if abs(val) > vabs * 0.6 else "black"
                ax.text(ri, 0, f"{val:+.3f}", ha="center", va="center",
                        fontsize=7, color=color)

    fig.colorbar(im, ax=axes.tolist(), label="Δη_c", shrink=0.8)
    plt.savefig("delta_eta_c.png", dpi=300, bbox_inches="tight")
    print("  저장: delta_eta_c.png")
    plt.close(fig)


# =========================================================================
# CSV 저장
# =========================================================================

def save_csv(results):
    rows = []
    for (model, fov, rho), r in sorted(results.items()):
        rows.append({
            "model": model,
            "fov": fov,
            "rho": rho,
            "N": r["N"],
            "L": r["L"],
            "v": r["v"],
            "eta_c": r["eta_c"],
            "chi_max": r["chi_max"],
        })
    df = pd.DataFrame(rows)
    df.to_csv("eta_c_summary.csv", index=False, float_format="%.4f")
    print("  저장: eta_c_summary.csv")


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Vicsek Model η_c 추출 (χ 최대값 기준)",
    )
    parser.add_argument("--folder", default="./Vicsek_Results")
    parser.add_argument("--v", type=float, default=None, help="v 값 필터")
    args = parser.parse_args()

    print("Loading data...")
    dfs = load_data(args.folder)
    print(f"  {len(dfs)} CSV files loaded.")
    if not dfs:
        return

    print("Calculating order parameters...")
    values = calculate_order_parameter(dfs)
    print(f"  {len(values)} parameter sets.")

    # v 자동 감지
    v_all = sorted(set(k[4] for k in values.keys()))
    v_filter = args.v if args.v is not None else v_all[0]
    print(f"  v = {v_filter}")

    print("Extracting η_c...")
    results, curves = extract_eta_c(values, v_filter)
    print(f"  {len(results)} (model, FOV, ρ) 조합.")

    print_summary(results)

    print("\nPlotting χ curves...")
    plot_chi_curves(results, curves)

    print("Plotting Δη_c heatmap...")
    plot_delta_eta_c(results)

    save_csv(results)
    print("\nDone.")


if __name__ == "__main__":
    main()