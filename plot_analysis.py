"""
plot_analysis.py — Order Parameter 통합 분석 그래프

Vicsek_Results + Vicsek_Results_Refined 데이터를 합쳐서
(FOV, ρ)마다 η별 φ(mean), χ(susceptibility), Binder cumulant를 그린다.

서로 다른 L(=N)은 선 스타일로 구분, metric/topologic은 색상으로 구분.
eta_c_summary.csv가 있으면 η_c 수직선 표시.

사용법:
  python plot_analysis.py                           # 전체
  python plot_analysis.py --fov 180 360             # 특정 FOV
  python plot_analysis.py --rho 0.5 1.0             # 특정 ρ
  python plot_analysis.py --quantity chi binder      # 특정 물리량만
  python plot_analysis.py --v 0.5                   # 특정 v
  python plot_analysis.py --dirs Vicsek_Results Vicsek_Results_Refined
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd


# =========================================================================
# 파일 파싱
# =========================================================================

def parse_filename(filename: str) -> dict | None:
    stem = os.path.basename(filename).replace(".csv", "")
    parts = stem.split("_")
    if len(parts) != 5:
        return None
    try:
        return {
            "fov": int(parts[0]), "L": float(parts[1]),
            "rho": float(parts[2]), "eta": float(parts[3]),
            "v": float(parts[4]),
        }
    except ValueError:
        return None


def load_all_csvs(dirs: list[str]) -> list[dict]:
    records = []
    for folder in dirs:
        if not os.path.isdir(folder):
            print(f"  [SKIP] 폴더 없음: {folder}")
            continue
        for path in glob.glob(os.path.join(folder, "*.csv")):
            meta = parse_filename(path)
            if meta is None:
                continue
            try:
                df = pd.read_csv(path)
                if len(df) == 0:
                    continue
            except Exception:
                continue
            meta["df"] = df
            records.append(meta)
    return records


# =========================================================================
# 물리량 계산
# =========================================================================

def calc_stats(df: pd.DataFrame, model: str, N: int) -> dict:
    prefix = "Metric" if model == "metric" else "Topologic"
    sub = df.filter(regex=prefix)
    if sub.empty:
        return {"mean": np.nan, "chi": np.nan, "binder": np.nan}

    phi_mean = sub.mean()
    phi2_mean = sub.pow(2).mean()
    phi4_mean = sub.pow(4).mean()

    return {
        "mean": float(phi_mean.mean()),
        "chi": float(N * (phi2_mean.mean() - phi_mean.pow(2).mean())),
        "binder": float(1 - (phi4_mean.mean() / (3 * phi2_mean.mean() ** 2))),
    }


# =========================================================================
# 시리즈 구축
# =========================================================================

def build_series(
    records: list[dict], v_filter: float,
) -> dict[tuple, pd.DataFrame]:
    """(model, fov, rho, L) → DataFrame[eta, mean, chi, binder, N]"""
    buf: dict[tuple, list] = {}

    for rec in records:
        if abs(rec["v"] - v_filter) > 1e-6:
            continue
        fov, L, rho, eta = rec["fov"], rec["L"], rec["rho"], rec["eta"]
        N = max(2, round(rho * L ** 2))

        for model in ["metric", "topologic"]:
            if rec["df"].filter(regex="Metric" if model == "metric" else "Topologic").empty:
                continue
            stats = calc_stats(rec["df"], model, N)
            key = (model, fov, rho, L)
            buf.setdefault(key, []).append({"eta": eta, "N": N, **stats})

    return {
        k: pd.DataFrame(v).sort_values("eta").reset_index(drop=True)
        for k, v in buf.items()
    }


# =========================================================================
# 스타일
# =========================================================================

MODEL_CMAPS = {"metric": plt.cm.Blues, "topologic": plt.cm.Reds}
MODEL_LABEL = {"metric": "Metric", "topologic": "Topologic"}

QUANTITY_INFO = {
    "mean":   {"ylabel": "⟨φ⟩",               "title": "Order Parameter"},
    "chi":    {"ylabel": "χ  (susceptibility)", "title": "Susceptibility"},
    "binder": {"ylabel": "G  (Binder)",         "title": "Binder Cumulant"},
}

_MARKERS = ["o", "s", "D", "^", "v", "P", "*"]
_LINES   = [":", "--", "-.", "-"]


def _style(L: float, L_list: list[float]) -> dict:
    rank = sorted(L_list).index(L)
    n = len(L_list)
    return dict(
        ls=_LINES[min(rank, 3)],
        marker=_MARKERS[rank % len(_MARKERS)],
        ms=3 + rank * 1.5,
        lw=0.8 + rank * 0.5,
        alpha=0.5 + 0.5 * (rank / max(1, n - 1)),
    )


# =========================================================================
# 그래프
# =========================================================================

def plot_quantities(
    series: dict[tuple, pd.DataFrame],
    quantities: list[str],
    eta_c_df: pd.DataFrame | None,
    fov_list: list[int],
    rho_list: list[float],
    v_val: float,
    output: str,
):
    n_qty = len(quantities)
    n_fov = len(fov_list)
    n_rho = len(rho_list)

    # 레이아웃: 행 = (ρ × quantity), 열 = FOV
    n_rows = n_rho * n_qty
    n_cols = n_fov

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 3 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    fig.suptitle(
        f"Vicsek Model Analysis  |  v={v_val}",
        fontsize=14, fontweight="bold",
    )

    models = sorted(set(k[0] for k in series.keys()))

    for ri, rho in enumerate(rho_list):
        for qi, qty in enumerate(quantities):
            row = ri * n_qty + qi
            info = QUANTITY_INFO[qty]

            for fi, fov in enumerate(fov_list):
                ax = axes[row, fi]
                has_data = False

                for model in models:
                    cmap = MODEL_CMAPS.get(model, plt.cm.Greys)

                    L_vals = sorted(
                        L for (m, f, r, L) in series
                        if m == model and f == fov and abs(r - rho) < 1e-6
                    )

                    for li, L in enumerate(L_vals):
                        sdf = series.get((model, fov, rho, L))
                        if sdf is None or sdf.empty:
                            continue

                        has_data = True
                        N = int(sdf["N"].iloc[0])
                        style = _style(L, L_vals)
                        color = cmap(0.4 + 0.5 * (li / max(1, len(L_vals) - 1)))

                        ax.plot(
                            sdf["eta"], sdf[qty],
                            color=color,
                            label=f"{MODEL_LABEL[model]} L={L:.0f} (N={N})",
                            **style,
                        )

                    # η_c 수직선
                    if eta_c_df is not None:
                        ec_row = eta_c_df[
                            (eta_c_df["model"] == model)
                            & (eta_c_df["fov"] == fov)
                            & ((eta_c_df["rho"] - rho).abs() < 1e-4)
                        ]
                        if not ec_row.empty and not ec_row["eta_c"].isna().all():
                            ec = float(ec_row["eta_c"].iloc[0])
                            ax.axvline(
                                ec, color=cmap(0.7), lw=1.0, ls="--", alpha=0.5,
                                label=f"η_c({MODEL_LABEL[model]})={ec:.3f}",
                            )

                # 장식
                if qty == "binder":
                    ax.axhline(0, color="gray", lw=0.6, alpha=0.5)

                ax.set_ylabel(info["ylabel"], fontsize=8)
                ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
                ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
                ax.grid(which="major", lw=0.5, alpha=0.3)
                ax.grid(which="minor", lw=0.3, alpha=0.15)

                # 제목: 맨 윗줄만
                if row == 0:
                    ax.set_title(f"FOV={fov}°", fontsize=10, fontweight="bold")

                # x라벨: 맨 아랫줄만
                if row == n_rows - 1:
                    ax.set_xlabel("η", fontsize=9)

                # ρ + quantity 표시: 왼쪽 끝 열만
                if fi == 0:
                    ax.annotate(
                        f"ρ={rho:.4g}\n{info['title']}",
                        xy=(0, 0.5), xycoords="axes fraction",
                        xytext=(-50, 0), textcoords="offset points",
                        fontsize=8, fontweight="bold",
                        ha="right", va="center", rotation=0,
                    )

                if has_data:
                    # 범례: 첫 번째 ρ의 첫 번째 quantity에만
                    if ri == 0 and qi == 0:
                        ax.legend(fontsize=5, loc="best", ncol=1)
                else:
                    ax.text(
                        0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="gray",
                    )

    plt.savefig(output, dpi=300, bbox_inches="tight")
    print(f"  저장: {output}")
    plt.close(fig)


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="φ, χ, Binder vs η 통합 분석 그래프",
    )
    parser.add_argument(
        "--dirs", nargs="+",
        default=["Vicsek_Results", "Vicsek_Results_Refined"],
    )
    parser.add_argument("--eta-csv", default="eta_c_summary.csv")
    parser.add_argument("--fov", type=int, nargs="+", default=None)
    parser.add_argument("--rho", type=float, nargs="+", default=None)
    parser.add_argument("--v", type=float, default=None)
    parser.add_argument(
        "--quantity", nargs="+", default=["mean", "chi", "binder"],
        choices=["mean", "chi", "binder"],
        help="표시할 물리량 (기본값: mean chi binder)",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print("Loading data...")
    records = load_all_csvs(args.dirs)
    print(f"  {len(records)} CSV files from {args.dirs}")
    if not records:
        print("[ERROR] CSV 없음")
        return

    v_all = sorted(set(r["v"] for r in records))
    v_val = args.v if args.v is not None else v_all[0]
    print(f"  v = {v_val}")

    series = build_series(records, v_val)
    del records
    print(f"  {len(series)} series.")

    if not series:
        print("[ERROR] 데이터 없음")
        return

    # 축 감지 + 필터
    all_fov = sorted(set(k[1] for k in series))
    all_rho = sorted(set(k[2] for k in series))
    all_L = sorted(set(k[3] for k in series))

    fov_list = [f for f in all_fov if args.fov is None or f in args.fov]
    rho_list = [
        r for r in all_rho
        if args.rho is None or any(abs(r - rf) < 1e-6 for rf in args.rho)
    ]

    # η_c CSV
    eta_c_df = None
    if os.path.exists(args.eta_csv):
        eta_c_df = pd.read_csv(args.eta_csv)
        if "v" in eta_c_df.columns:
            eta_c_df = eta_c_df[abs(eta_c_df["v"] - v_val) < 1e-6]
        print(f"  η_c: {len(eta_c_df)} entries")

    print(f"  FOV: {fov_list}  |  ρ: {rho_list}  |  L: {all_L}")
    print(f"  물리량: {args.quantity}")

    output = args.output or "analysis.png"

    print("Drawing...")
    plot_quantities(
        series=series,
        quantities=args.quantity,
        eta_c_df=eta_c_df,
        fov_list=fov_list,
        rho_list=rho_list,
        v_val=v_val,
        output=output,
    )
    print("Done.")


if __name__ == "__main__":
    main()