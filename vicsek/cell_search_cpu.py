"""
Vicsek Model — 셀 리스트 기반 이웃 탐색 (CPU 전용)

Topologic(k-NN) 및 Metric 모델의 셀 리스트 탐색.
numba 설치 시 JIT 병렬화, 미설치 시 NumPy 폴백.
"""
from __future__ import annotations

import numpy as np

# =========================================================================
# Topologic(k-NN) 셀 탐색: NumPy 폴백
# =========================================================================

def _topologic_cell_numpy(
    px, py, cos_t, sin_t, theta,
    sort_idx, cell_start,
    N, n_cells_1d, cs, L,
    k, n_rings, use_fov, cos_half,
):
    offsets = [
        (dr, dc)
        for dr in range(-n_rings, n_rings + 1)
        for dc in range(-n_rings, n_rings + 1)
    ]
    cx_all = np.minimum((px / cs).astype(np.int32), n_cells_1d - 1)
    cy_all = np.minimum((py / cs).astype(np.int32), n_cells_1d - 1)

    avg_per_cell = max(1, int(N / (n_cells_1d * n_cells_1d) + 1))
    max_cand = min(N - 1, len(offsets) * (avg_per_cell + 4))
    max_cand = max(max_cand, k + 1)

    cand_d = np.full((N, max_cand), np.inf, dtype=np.float32)
    cand_i = np.zeros((N, max_cand), dtype=np.int32)
    fill = np.zeros(N, dtype=np.int32)
    arange_N = np.arange(N, dtype=np.int32)

    for dr, dc in offsets:
        ncy = (cy_all + dr) % n_cells_1d
        ncx = (cx_all + dc) % n_cells_1d
        nc = ncy * n_cells_1d + ncx
        c_start = cell_start[nc]
        c_end = cell_start[nc + 1]
        max_in = int((c_end - c_start).max())
        if max_in <= 0:
            continue
        for s in range(max_in):
            p_idx = np.minimum(c_start + s, N - 1)
            in_cell = (c_start + s) < c_end
            j = sort_idx[p_idx]
            valid = in_cell & (j != arange_N)
            if not valid.any():
                continue
            ddx = px - px[j]; ddy = py - py[j]
            ddx -= L * np.round(ddx / L)
            ddy -= L * np.round(ddy / L)
            d2 = ddx * ddx + ddy * ddy
            if use_fov:
                norm = np.sqrt(np.maximum(d2, 1e-18))
                dot = np.clip(cos_t * (ddx / norm) + sin_t * (ddy / norm), -1.0, 1.0)
                valid = valid & (dot >= cos_half)
            row = np.where(valid & (fill < max_cand))[0]
            if row.size == 0:
                continue
            col = fill[row]
            cand_d[row, col] = d2[row]
            cand_i[row, col] = j[row]
            fill[row] = np.minimum(fill[row] + 1, max_cand - 1)

    k_actual = min(k, max_cand)
    part = np.argpartition(cand_d, k_actual, axis=1)[:, :k_actual]
    chosen_d = np.take_along_axis(cand_d, part, axis=1)
    chosen_i = np.take_along_axis(cand_i, part, axis=1)
    valid_k = chosen_d != np.inf

    sc = np.where(valid_k, cos_t[chosen_i], 0.0).sum(axis=1)
    ss = np.where(valid_k, sin_t[chosen_i], 0.0).sum(axis=1)
    n_v = valid_k.sum(axis=1).astype(np.float32)
    safe_n = np.where(n_v > 0, n_v, 1.0)
    has_nb = n_v > 0
    return (
        np.where(has_nb, sc / safe_n, cos_t),
        np.where(has_nb, ss / safe_n, sin_t),
    )


# =========================================================================
# Metric 셀 탐색: NumPy 폴백
# =========================================================================

def _metric_cell_numpy(
    px, py, cos_t, sin_t,
    sort_idx, cell_start,
    N, n_cells_1d, cs, L, r_sq,
    use_fov, cos_half,
):
    cx_all = np.minimum((px / cs).astype(np.int32), n_cells_1d - 1)
    cy_all = np.minimum((py / cs).astype(np.int32), n_cells_1d - 1)
    arange_N = np.arange(N, dtype=np.int32)
    sc_acc = np.zeros(N, dtype=np.float32)
    ss_acc = np.zeros(N, dtype=np.float32)
    cnt = np.zeros(N, dtype=np.int32)

    for dr in range(-1, 2):
        for dc in range(-1, 2):
            ncy = (cy_all + dr) % n_cells_1d
            ncx = (cx_all + dc) % n_cells_1d
            nc = ncy * n_cells_1d + ncx
            c_start = cell_start[nc]
            c_end = cell_start[nc + 1]
            max_in = int((c_end - c_start).max())
            if max_in <= 0:
                continue
            for s in range(max_in):
                p_idx = np.minimum(c_start + s, N - 1)
                in_cell = (c_start + s) < c_end
                j = sort_idx[p_idx]
                valid = in_cell & (j != arange_N)
                if not valid.any():
                    continue
                ddx = px - px[j]; ddy = py - py[j]
                ddx -= L * np.round(ddx / L)
                ddy -= L * np.round(ddy / L)
                d2 = ddx * ddx + ddy * ddy
                valid = valid & (d2 < r_sq)
                if use_fov:
                    norm = np.sqrt(np.maximum(d2, 1e-18))
                    dot = np.clip(cos_t * (ddx / norm) + sin_t * (ddy / norm), -1.0, 1.0)
                    valid = valid & (dot >= cos_half)
                sc_acc += np.where(valid, cos_t[j], 0.0)
                ss_acc += np.where(valid, sin_t[j], 0.0)
                cnt += valid.astype(np.int32)

    has_nb = cnt > 0
    safe_n = np.where(has_nb, cnt, 1).astype(np.float32)
    return (
        np.where(has_nb, sc_acc / safe_n, cos_t),
        np.where(has_nb, ss_acc / safe_n, sin_t),
    )


# =========================================================================
# numba JIT 버전
# =========================================================================

_NUMBA_TOPOLOGIC = False
_NUMBA_METRIC = False

try:
    from numba import njit, prange

    @njit(parallel=True, cache=True, fastmath=True)
    def _topologic_cell_numba(
        px, py, cos_t, sin_t, theta,
        sort_idx, cell_start,
        N, n_cells_1d, cs, L,
        k, n_rings, use_fov, cos_half,
    ):
        avg_cos = np.empty(N, dtype=np.float32)
        avg_sin = np.empty(N, dtype=np.float32)
        for i in prange(N):
            xi = px[i]; yi = py[i]; ci = cos_t[i]; si = sin_t[i]
            cxi = min(int(xi / cs), n_cells_1d - 1)
            cyi = min(int(yi / cs), n_cells_1d - 1)
            KMAX = 64
            heap_d = np.full(KMAX, np.inf, dtype=np.float32)
            heap_i = np.zeros(KMAX, dtype=np.int32)
            heap_sz = 0; heap_max = 0.0; heap_mpos = 0
            for dr in range(-n_rings, n_rings + 1):
                for dc in range(-n_rings, n_rings + 1):
                    ncy = (cyi + dr) % n_cells_1d
                    ncx = (cxi + dc) % n_cells_1d
                    nc = ncy * n_cells_1d + ncx
                    s = cell_start[nc]; e = cell_start[nc + 1]
                    for p in range(s, e):
                        j = sort_idx[p]
                        if j == i: continue
                        ddx = xi - px[j]; ddy = yi - py[j]
                        ddx -= L * round(ddx / L); ddy -= L * round(ddy / L)
                        d2 = ddx * ddx + ddy * ddy
                        if use_fov:
                            norm = (d2 + 1e-18) ** 0.5
                            dot = ci * ddx / norm + si * ddy / norm
                            if dot < -1.0: dot = -1.0
                            if dot > 1.0: dot = 1.0
                            if dot < cos_half: continue
                        if heap_sz < k:
                            heap_d[heap_sz] = d2; heap_i[heap_sz] = j; heap_sz += 1
                            if heap_sz == 1 or d2 > heap_max: heap_max = d2; heap_mpos = heap_sz - 1
                        elif d2 < heap_max:
                            heap_d[heap_mpos] = d2; heap_i[heap_mpos] = j
                            heap_max = heap_d[0]; heap_mpos = 0
                            for q in range(1, k):
                                if heap_d[q] > heap_max: heap_max = heap_d[q]; heap_mpos = q
            sc = 0.0; ss = 0.0
            for q in range(heap_sz): sc += cos_t[heap_i[q]]; ss += sin_t[heap_i[q]]
            if heap_sz > 0: avg_cos[i] = sc / heap_sz; avg_sin[i] = ss / heap_sz
            else: avg_cos[i] = ci; avg_sin[i] = si
        return avg_cos, avg_sin

    _NUMBA_TOPOLOGIC = True
except ImportError:
    pass

try:
    from numba import njit as _njit, prange as _prange

    @_njit(parallel=True, cache=True, fastmath=True)
    def _metric_cell_numba(
        px, py, cos_t, sin_t,
        sort_idx, cell_start,
        N, n_cells_1d, cs, L, r_sq,
        use_fov, cos_half,
    ):
        avg_cos = np.empty(N, dtype=np.float32)
        avg_sin = np.empty(N, dtype=np.float32)
        for i in _prange(N):
            xi = px[i]; yi = py[i]; ci = cos_t[i]; si = sin_t[i]
            cxi = min(int(xi / cs), n_cells_1d - 1)
            cyi = min(int(yi / cs), n_cells_1d - 1)
            sc = 0.0; ss = 0.0; cnt = 0
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    ncy = (cyi + dr) % n_cells_1d; ncx = (cxi + dc) % n_cells_1d
                    nc = ncy * n_cells_1d + ncx
                    s = cell_start[nc]; e = cell_start[nc + 1]
                    for p in range(s, e):
                        j = sort_idx[p]
                        if j == i: continue
                        ddx = xi - px[j]; ddy = yi - py[j]
                        ddx -= L * round(ddx / L); ddy -= L * round(ddy / L)
                        d2 = ddx * ddx + ddy * ddy
                        if d2 >= r_sq: continue
                        if use_fov:
                            norm = (d2 + 1e-18) ** 0.5
                            dot = ci * ddx / norm + si * ddy / norm
                            if dot < -1.0: dot = -1.0
                            if dot > 1.0: dot = 1.0
                            if dot < cos_half: continue
                        sc += cos_t[j]; ss += sin_t[j]; cnt += 1
            if cnt > 0: avg_cos[i] = sc / cnt; avg_sin[i] = ss / cnt
            else: avg_cos[i] = ci; avg_sin[i] = si
        return avg_cos, avg_sin

    _NUMBA_METRIC = True
except (ImportError, Exception):
    pass


# =========================================================================
# 디스패치 함수
# =========================================================================

def topologic_cell_search(
    px, py, cos_t, sin_t, theta,
    sort_idx, cell_start,
    N, n_cells_1d, cs, L,
    k, n_rings, use_fov, cos_half,
):
    if _NUMBA_TOPOLOGIC:
        return _topologic_cell_numba(
            px, py, cos_t, sin_t, theta, sort_idx, cell_start,
            N, n_cells_1d, np.float32(cs), np.float32(L),
            k, n_rings, use_fov, np.float32(cos_half),
        )
    return _topologic_cell_numpy(
        px, py, cos_t, sin_t, theta, sort_idx, cell_start,
        N, n_cells_1d, cs, L, k, n_rings, use_fov, cos_half,
    )


def metric_cell_search(
    px, py, cos_t, sin_t,
    sort_idx, cell_start,
    N, n_cells_1d, cs, L, r_sq,
    use_fov, cos_half,
):
    if _NUMBA_METRIC:
        return _metric_cell_numba(
            px, py, cos_t, sin_t, sort_idx, cell_start,
            N, n_cells_1d, np.float32(cs), np.float32(L), np.float32(r_sq),
            use_fov, np.float32(cos_half),
        )
    return _metric_cell_numpy(
        px, py, cos_t, sin_t, sort_idx, cell_start,
        N, n_cells_1d, cs, L, r_sq, use_fov, cos_half,
    )