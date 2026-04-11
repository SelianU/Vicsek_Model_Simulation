"""
Vicsek Model — Order Parameter 분석 및 히트맵 생성

새 파일 형식 대응:
  - CSV 파일명: {fov}_{L}_{rho}_{eta}_{v}.csv
  - 컬럼명: Metric_Trial_*, Topologic_Trial_*
  - η ∈ [0, 1] 정규화 (변환 불필요)
  - 파라미터를 파일명에서 자동 감지 (하드코딩 불필요)

사용법:
  python order_parameter.py                          # 기본 폴더
  python order_parameter.py --folder Vicsek_Results  # 폴더 지정
  python order_parameter.py --v 0.5                  # 특정 v만 필터
"""
from __future__ import annotations

import argparse
import os
import glob
from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================================================================
# 데이터 로드
# =========================================================================

@dataclass(frozen=True)
class SimKey:
    """CSV 파일 하나의 파라미터 조합."""
    fov: int
    L: float
    rho: float
    eta: float
    v: float

    @property
    def N(self) -> int:
        return max(2, round(self.rho * self.L ** 2))


def parse_filename(filename: str) -> SimKey | None:
    """파일명에서 파라미터를 추출한다.

    형식: {fov}_{L}_{rho}_{eta}_{v}.csv
    """
    stem = os.path.basename(filename).replace(".csv", "")
    parts = stem.split("_")
    if len(parts) != 5:
        return None
    try:
        return SimKey(
            fov=int(parts[0]),
            L=float(parts[1]),
            rho=float(parts[2]),
            eta=float(parts[3]),
            v=float(parts[4]),
        )
    except ValueError:
        return None


def load_data(folder: str) -> dict[SimKey, pd.DataFrame]:
    """폴더 내 모든 CSV를 파싱하여 {SimKey: DataFrame} 반환."""
    pattern = os.path.join(folder, "*.csv")
    file_list = glob.glob(pattern)

    dfs = {}
    for path in file_list:
        key = parse_filename(path)
        if key is None:
            continue
        try:
            df = pd.read_csv(path)
            if len(df) > 0:
                dfs[key] = df
        except Exception:
            continue

    return dfs


# =========================================================================
# Order Parameter 계산
# =========================================================================

METRICS = [
    "metric_mean", "metric_chi", "metric_binder",
    "topologic_mean", "topologic_chi", "topologic_binder",
]


def calculate_order_parameter(
    dfs: dict[SimKey, pd.DataFrame],
) -> dict[SimKey, dict[str, float]]:
    """각 파라미터 조합에 대해 <φ>, χ, Binder cumulant를 계산."""
    values = {}

    for key, df in dfs.items():
        N = key.N
        result = {}

        for model_prefix, label in [("Metric", "metric"), ("Topologic", "topologic")]:
            cols = df.filter(regex=model_prefix)
            if cols.empty:
                continue

            # trial별 시간 평균 → trial 앙상블
            phi_mean = cols.mean()           # 각 trial의 <φ>_t
            phi2_mean = cols.pow(2).mean()   # <φ²>_t
            phi4_mean = cols.pow(4).mean()   # <φ⁴>_t

            mean_phi = phi_mean.mean()                           # <<φ>>
            chi = N * (phi2_mean.mean() - phi_mean.pow(2).mean())  # N(<φ²> - <φ>²)
            binder = 1 - (phi4_mean.mean() / (3 * phi2_mean.mean() ** 2))

            result[f"{label}_mean"] = mean_phi
            result[f"{label}_chi"] = chi
            result[f"{label}_binder"] = binder

        if result:
            values[key] = result

    return values


# =========================================================================
# 파라미터 축 자동 감지
# =========================================================================

def detect_axes(keys: list[SimKey]) -> dict:
    """데이터에 존재하는 고유 파라미터 값들을 정렬하여 반환."""
    return {
        "fov": sorted(set(k.fov for k in keys)),
        "rho": sorted(set(k.rho for k in keys)),
        "eta": sorted(set(k.eta for k in keys)),
        "v":   sorted(set(k.v for k in keys)),
        "L":   sorted(set(k.L for k in keys)),
    }


# =========================================================================
# 히트맵 그리기
# =========================================================================

def draw_heatmap(
    data: dict[SimKey, dict[str, float]],
    v_filter: float | None = None,
    output: str = "heatmap.png",
):
    """ρ × η 히트맵을 FOV별 열, metric별 행으로 그린다.

    Args:
        v_filter: 특정 v 값만 필터. None이면 첫 번째 v 사용.
        output: 저장 파일명.
    """
    axes_info = detect_axes(list(data.keys()))

    # v 필터링
    if v_filter is not None:
        v_val = v_filter
    else:
        v_val = axes_info["v"][0]
    print(f"  v = {v_val} 기준으로 히트맵 생성")

    filtered = {k: v for k, v in data.items() if abs(k.v - v_val) < 1e-6}
    if not filtered:
        print("[ERROR] 해당 v 값에 대한 데이터가 없습니다.")
        return

    axes_info = detect_axes(list(filtered.keys()))
    FOV_list = axes_info["fov"]
    RHO_list = axes_info["rho"]
    ETA_list = axes_info["eta"]
    L_val = axes_info["L"][0]

    # 실제 존재하는 metric만 사용
    available_metrics = set()
    for v in filtered.values():
        available_metrics.update(v.keys())
    active_metrics = [m for m in METRICS if m in available_metrics]

    if not active_metrics:
        print("[ERROR] 계산된 metric이 없습니다.")
        return

    n_rows = len(active_metrics)
    n_cols = len(FOV_list)

    fig, axes_arr = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3.5 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    fig.suptitle(
        f"Vicsek Model  |  L={L_val}  v={v_val}",
        fontsize=14, fontweight="bold",
    )

    for mi, metric in enumerate(active_metrics):
        for fi, fov in enumerate(FOV_list):
            ax = axes_arr[mi, fi]
            grid = np.full((len(ETA_list), len(RHO_list)), np.nan)

            for ei, eta in enumerate(ETA_list):
                for ri, rho in enumerate(RHO_list):
                    entry = filtered.get(SimKey(
                        fov=fov, L=L_val, rho=rho, eta=eta, v=v_val,
                    ))
                    if entry is not None and metric in entry:
                        grid[ei, ri] = entry[metric]

            if np.all(np.isnan(grid)):
                ax.set_visible(False)
                continue

            vmin = np.nanmin(grid)
            vmax = np.nanmax(grid)
            if vmin == vmax:
                vmax = vmin + 1e-6

            im = ax.imshow(
                grid, cmap="YlOrRd", vmin=vmin, vmax=vmax,
                aspect="auto", origin="lower",
            )

            # X축: 밀도 (ρ) → N 변환도 표시
            ax.set_xticks(range(len(RHO_list)))
            N_labels = [f"{r}\n(N={max(2, round(r * L_val**2))})"
                        for r in RHO_list]
            ax.set_xticklabels(N_labels, fontsize=5.5, ha="center")
            ax.set_xlabel("ρ  (N)", fontsize=8)

            # Y축: η (0~1)
            ax.set_yticks(range(len(ETA_list)))
            ax.set_yticklabels([f"{e:.2f}" for e in ETA_list], fontsize=6)
            ax.set_ylabel("η", fontsize=8)

            # 제목
            model_label = metric.split("_")[0].capitalize()
            stat_label = metric.split("_", 1)[1]
            ax.set_title(
                f"{model_label} {stat_label}  |  FOV={fov}°",
                fontsize=10, fontweight="bold",
            )

            # 셀 값 텍스트
            for ei in range(len(ETA_list)):
                for ri in range(len(RHO_list)):
                    val = grid[ei, ri]
                    if not np.isnan(val):
                        color = "white" if val > (vmin + vmax) / 2 else "black"
                        ax.text(
                            ri, ei, f"{val:.3f}",
                            ha="center", va="center",
                            fontsize=5, color=color,
                        )

        # 행별 컬러바
        fig.colorbar(
            im, ax=axes_arr[mi, :].tolist(),
            label=metric, shrink=0.6,
        )

    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"  저장: {output}")
    plt.close(fig)


# =========================================================================
# 라인 플롯 (η에 따른 φ 변화, N별 비교)
# =========================================================================

def draw_line_plot(
    data: dict[SimKey, dict[str, float]],
    v_filter: float | None = None,
    output: str = "order_parameter.png",
):
    """각 FOV × 모델에 대해 η-φ 라인 플롯을 그린다.

    N(밀도)별로 다른 색상의 라인으로 표시.
    """
    axes_info = detect_axes(list(data.keys()))

    v_val = v_filter if v_filter is not None else axes_info["v"][0]
    filtered = {k: v for k, v in data.items() if abs(k.v - v_val) < 1e-6}
    if not filtered:
        print("[ERROR] 해당 v 값에 대한 데이터가 없습니다.")
        return

    axes_info = detect_axes(list(filtered.keys()))
    FOV_list = axes_info["fov"]
    RHO_list = axes_info["rho"]
    ETA_list = axes_info["eta"]
    L_val = axes_info["L"][0]

    # 모델 감지
    models = []
    sample = next(iter(filtered.values()))
    if "metric_mean" in sample:
        models.append(("metric_mean", "Metric"))
    if "topologic_mean" in sample:
        models.append(("topologic_mean", "Topologic"))

    if not models:
        print("[ERROR] mean 데이터 없음")
        return

    n_rows = len(models)
    n_cols = len(FOV_list)
    fig, axes_arr = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    fig.suptitle(
        f"Order Parameter φ vs η  |  L={L_val}  v={v_val}",
        fontsize=14, fontweight="bold",
    )

    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, len(RHO_list) - 1)) for i in range(len(RHO_list))]

    for mi, (metric_key, model_name) in enumerate(models):
        for fi, fov in enumerate(FOV_list):
            ax = axes_arr[mi, fi]

            for ri, rho in enumerate(RHO_list):
                phis = []
                etas = []
                for eta in ETA_list:
                    entry = filtered.get(SimKey(
                        fov=fov, L=L_val, rho=rho, eta=eta, v=v_val,
                    ))
                    if entry and metric_key in entry:
                        etas.append(eta)
                        phis.append(entry[metric_key])

                if etas:
                    N_val = max(2, round(rho * L_val ** 2))
                    ax.plot(
                        etas, phis,
                        "o-", color=colors[ri], markersize=3, linewidth=1.2,
                        label=f"ρ={rho} (N={N_val})",
                    )

            ax.set_xlabel("η", fontsize=9)
            ax.set_ylabel("φ", fontsize=9)
            ax.set_title(f"{model_name}  |  FOV={fov}°", fontsize=10, fontweight="bold")
            ax.set_xlim(-0.02, 1.02)
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)

            if fi == n_cols - 1:
                ax.legend(fontsize=6, loc="upper right")

    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"  저장: {output}")
    plt.close(fig)


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Vicsek Model Order Parameter 분석 및 시각화",
    )
    parser.add_argument(
        "--folder", type=str, default="./Vicsek_Results",
        help="CSV 폴더 경로 (기본값: ./Vicsek_Results)",
    )
    parser.add_argument(
        "--v", type=float, default=None,
        help="특정 v 값만 필터 (기본값: 첫 번째 v)",
    )
    parser.add_argument(
        "--no-heatmap", action="store_true",
        help="히트맵 생성 안 함",
    )
    parser.add_argument(
        "--no-line", action="store_true",
        help="라인 플롯 생성 안 함",
    )
    parser.add_argument(
        "--dpi", type=int, default=300,
        help="출력 DPI (기본값: 300)",
    )
    args = parser.parse_args()

    print("Loading data...")
    dfs = load_data(args.folder)
    print(f"  {len(dfs)} CSV files loaded.")

    if not dfs:
        print("[ERROR] CSV 파일을 찾을 수 없습니다.")
        return

    # 파라미터 요약
    axes_info = detect_axes(list(dfs.keys()))
    print(f"  FOV:  {axes_info['fov']}")
    print(f"  ρ:    {axes_info['rho']}")
    print(f"  η:    {[round(e, 4) for e in axes_info['eta']]}")
    print(f"  v:    {axes_info['v']}")
    print(f"  L:    {axes_info['L']}")

    print("Calculating order parameters...")
    order_params = calculate_order_parameter(dfs)
    print(f"  {len(order_params)} parameter sets calculated.")

    if not args.no_heatmap:
        print("Drawing heatmap...")
        draw_heatmap(order_params, v_filter=args.v)

    if not args.no_line:
        print("Drawing line plot...")
        draw_line_plot(order_params, v_filter=args.v)

    print("Done.")


if __name__ == "__main__":
    main()