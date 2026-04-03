"""
Vicsek Model — GPU 물리 엔진 (CuPy + CUDA)

변경사항: step_auto (#2), CELL_THRESHOLD (#3), topologic (#11)
"""
from __future__ import annotations

import sys

import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

from .config import SimConfig
from .convergence import compute_n_rings

# CUDA 커널 소스 (내부 함수명은 유지, Python API만 topologic으로 변경)
_CELL_KERNEL_SRC = r"""
extern "C" {
__global__ void assign_cells(
    const float* __restrict__ px, const float* __restrict__ py,
    int* __restrict__ cell_id, int N, int n_cells_1d, float cs
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int cx = min((int)(px[i] / cs), n_cells_1d - 1);
    int cy = min((int)(py[i] / cs), n_cells_1d - 1);
    cell_id[i] = cy * n_cells_1d + cx;
}
__global__ void knn_cell_avg(
    const float* __restrict__ px, const float* __restrict__ py,
    const float* __restrict__ cos_t, const float* __restrict__ sin_t,
    const float* __restrict__ theta, const int* __restrict__ sort_idx,
    const int* __restrict__ cell_start, float* __restrict__ avg_cos,
    float* __restrict__ avg_sin, int N, int n_cells_1d, float cs, float L,
    int k, int n_rings, int use_fov, float cos_half_fov
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float xi=px[i],yi=py[i],ci=cos_t[i],si=sin_t[i];
    int cxi=min((int)(xi/cs),n_cells_1d-1), cyi=min((int)(yi/cs),n_cells_1d-1);
    const int KMAX=64; float heap_d[KMAX]; int heap_i[KMAX];
    int heap_sz=0; float heap_max=0.0f; int heap_max_pos=0;
    for(int dr=-n_rings;dr<=n_rings;dr++){for(int dc=-n_rings;dc<=n_rings;dc++){
        int ncy=((cyi+dr)%n_cells_1d+n_cells_1d)%n_cells_1d;
        int ncx=((cxi+dc)%n_cells_1d+n_cells_1d)%n_cells_1d;
        int nc=ncy*n_cells_1d+ncx, s=cell_start[nc], e=cell_start[nc+1];
        for(int p=s;p<e;p++){int j=sort_idx[p]; if(j==i) continue;
            float ddx=xi-px[j],ddy=yi-py[j];
            ddx-=L*rintf(ddx/L); ddy-=L*rintf(ddy/L);
            float d2=ddx*ddx+ddy*ddy;
            if(use_fov){float norm=sqrtf(fmaxf(d2,1e-18f));
                float dot=fmaf(ci,ddx/norm,si*(ddy/norm));
                dot=fminf(fmaxf(dot,-1.0f),1.0f); if(dot<cos_half_fov) continue;}
            if(heap_sz<k){heap_d[heap_sz]=d2;heap_i[heap_sz]=j;heap_sz++;
                if(heap_sz==1||d2>heap_max){heap_max=d2;heap_max_pos=heap_sz-1;}
            }else if(d2<heap_max){heap_d[heap_max_pos]=d2;heap_i[heap_max_pos]=j;
                heap_max=heap_d[0];heap_max_pos=0;
                for(int q=1;q<k;q++){if(heap_d[q]>heap_max){heap_max=heap_d[q];heap_max_pos=q;}}}}}}
    float sc=0.0f,ss=0.0f;
    for(int q=0;q<heap_sz;q++){sc+=cos_t[heap_i[q]];ss+=sin_t[heap_i[q]];}
    if(heap_sz>0){avg_cos[i]=sc/heap_sz;avg_sin[i]=ss/heap_sz;}
    else{avg_cos[i]=ci;avg_sin[i]=si;}
}
__global__ void metric_cell_avg(
    const float* __restrict__ px, const float* __restrict__ py,
    const float* __restrict__ cos_t, const float* __restrict__ sin_t,
    const float* __restrict__ theta, const int* __restrict__ sort_idx,
    const int* __restrict__ cell_start, float* __restrict__ avg_cos,
    float* __restrict__ avg_sin, int N, int n_cells_1d, float cs, float L,
    float r_sq, int use_fov, float cos_half_fov
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float xi=px[i],yi=py[i],ci=cos_t[i],si=sin_t[i];
    int cxi=min((int)(xi/cs),n_cells_1d-1), cyi=min((int)(yi/cs),n_cells_1d-1);
    float sc=0.0f,ss=0.0f; int cnt=0;
    for(int dr=-1;dr<=1;dr++){for(int dc=-1;dc<=1;dc++){
        int ncy=((cyi+dr)%n_cells_1d+n_cells_1d)%n_cells_1d;
        int ncx=((cxi+dc)%n_cells_1d+n_cells_1d)%n_cells_1d;
        int nc=ncy*n_cells_1d+ncx, s=cell_start[nc], e=cell_start[nc+1];
        for(int p=s;p<e;p++){int j=sort_idx[p]; if(j==i) continue;
            float ddx=xi-px[j],ddy=yi-py[j];
            ddx-=L*rintf(ddx/L); ddy-=L*rintf(ddy/L);
            float d2=ddx*ddx+ddy*ddy; if(d2>=r_sq) continue;
            if(use_fov){float norm=sqrtf(fmaxf(d2,1e-18f));
                float dot=fmaf(ci,ddx/norm,si*(ddy/norm));
                dot=fminf(fmaxf(dot,-1.0f),1.0f); if(dot<cos_half_fov) continue;}
            sc+=cos_t[j];ss+=sin_t[j];cnt+=1;}}}
    if(cnt>0){avg_cos[i]=sc/cnt;avg_sin[i]=ss/cnt;}
    else{avg_cos[i]=ci;avg_sin[i]=si;}
}
/*
 * backpropagate_cells - empty cell backfill (GPU-side)
 *
 * Replaces the CPU-side cell_start.get() + Python loop.
 * Scans backward: if cell_start[c] == N (empty), copy cell_start[c+1].
 * Single-thread kernel (n_cells is O(thousands), fast enough).
 */
__global__ void backpropagate_cells(
    int* __restrict__ cell_start,
    int n_cells,
    int N
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int c = n_cells - 1; c >= 0; c--) {
        if (cell_start[c] == N) {
            cell_start[c] = cell_start[c + 1];
        }
    }
}

} /* extern "C" */
"""


class VicsekSimulatorGPU:
    """GPU(CuPy + CUDA) 기반 Vicsek 모델 물리 엔진."""

    CELL_THRESHOLD = 1000  # (#3)
    _cell_module = None

    def __init__(self, cfg: SimConfig):
        if cp is None:
            sys.exit("CuPy 미설치. GPU 모드 불가.")
        self.cfg = cfg
        self._init_gpu()

    def _init_gpu(self):
        try:
            cp.cuda.Device().use()
            name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
            print(f"GPU: {name}  |  dtype: {self.cfg.dtype.__name__}")
        except Exception as e:
            sys.exit(f"CUDA error: {e}")

    # ------------------------------------------------------------------
    # 자동 라우팅 (#2)
    # ------------------------------------------------------------------

    def step_auto(self, model, pos, theta, noise, N, fov_rad):
        if model == "metric":
            if N >= self.CELL_THRESHOLD:
                return self.step_metric_cell(pos, theta, noise, N, fov_rad)
            return self.step_metric(pos, theta, noise, N, fov_rad)
        else:
            k = min(self.cfg.k_topologic, N - 1) if N > 1 else 0
            if N >= self.CELL_THRESHOLD:
                rho = N / (self.cfg.L ** 2)
                n_rings = compute_n_rings(k, rho, fov_rad, self.cfg._two_pi)
                return self.step_topologic_cell(
                    pos, theta, noise, N, fov_rad, k, n_rings, 1.0
                )
            return self.step_topologic(pos, theta, noise, N, fov_rad, k)

    # ------------------------------------------------------------------
    # Metric / Topologic 스텝 (행렬 기반)
    # ------------------------------------------------------------------

    def step_metric(self, pos, theta, noise, N, fov_rad):
        cfg = self.cfg; cos_t = cp.cos(theta); sin_t = cp.sin(theta)
        dx, dy = self._pbc(pos); dist_sq = dx * dx + dy * dy
        if fov_rad >= cfg._two_pi - cfg._eps_fov or N <= 1:
            nb = (dist_sq < cfg._r_sq) & (dist_sq > cfg._eps_sq)
            cp.fill_diagonal(nb, False)
        else:
            nb = self._metric_fov(cos_t, sin_t, dx, dy, dist_sq, fov_rad)
        avg = self._wavg(theta, cos_t, sin_t, nb, cfg.dtype)
        theta[:] = avg + noise
        return (*self._move(pos, theta, N), )

    def step_topologic(self, pos, theta, noise, N, fov_rad, k):
        cfg = self.cfg; cos_t = cp.cos(theta); sin_t = cp.sin(theta)
        dx, dy = self._pbc(pos); dist_sq = dx * dx + dy * dy
        cp.fill_diagonal(dist_sq, cp.inf)
        if fov_rad < cfg._two_pi - cfg._eps_fov and N > 1:
            dist_sq = self._fov_filter(cos_t, sin_t, dx, dy, dist_sq, fov_rad)
        part = cp.argpartition(dist_sq, k, axis=1)[:, :k]
        chosen_d = cp.take_along_axis(dist_sq, part, axis=1)
        valid = chosen_d != cp.inf
        avg = self._tavg(theta, cos_t, sin_t, part, valid, cfg.dtype)
        theta[:] = avg + noise
        return (*self._move(pos, theta, N), )

    # ------------------------------------------------------------------
    # 셀 리스트 CUDA 커널 기반 스텝
    # ------------------------------------------------------------------

    def _get_mod(self):
        if VicsekSimulatorGPU._cell_module is None:
            VicsekSimulatorGPU._cell_module = cp.RawModule(
                code=_CELL_KERNEL_SRC, options=("--std=c++14",)
            )
        return VicsekSimulatorGPU._cell_module

    def step_topologic_cell(self, pos, theta, noise, N, fov_rad, k, n_rings, cell_size):
        cfg = self.cfg; mod = self._get_mod()
        n_cells_1d = max(1, int(cfg.L / cell_size))
        cs = cp.float32(cfg.L / n_cells_1d)
        BLOCK = 256; grid = (N + BLOCK - 1) // BLOCK
        px = cp.ascontiguousarray(pos[:, 0]); py = cp.ascontiguousarray(pos[:, 1])
        cell_id = cp.empty(N, dtype=cp.int32)
        mod.get_function("assign_cells")((grid,),(BLOCK,),(px,py,cell_id,cp.int32(N),cp.int32(n_cells_1d),cs))
        si, cst = self._cell_list_gpu(cell_id, n_cells_1d * n_cells_1d, N)
        cos_t = cp.cos(theta); sin_t = cp.sin(theta)
        ac = cp.empty(N, dtype=cp.float32); asin = cp.empty(N, dtype=cp.float32)
        uf = int(fov_rad < cfg._two_pi - cfg._eps_fov and N > 1)
        ch = cp.float32(cfg.get_cos_half(fov_rad))
        mod.get_function("knn_cell_avg")(
            (grid,),(BLOCK,),
            (px,py,cos_t,sin_t,theta,si,cst,ac,asin,
             cp.int32(N),cp.int32(n_cells_1d),cs,cp.float32(cfg.L),
             cp.int32(k),cp.int32(n_rings),cp.int32(uf),ch))
        theta[:] = cp.arctan2(asin, ac) + noise
        return (*self._move(pos, theta, N), )

    def step_metric_cell(self, pos, theta, noise, N, fov_rad):
        cfg = self.cfg; mod = self._get_mod()
        cs_f = cfg.r_metric; n_cells_1d = max(1, int(cfg.L / cs_f))
        cs = cp.float32(cfg.L / n_cells_1d)
        BLOCK = 256; grid = (N + BLOCK - 1) // BLOCK
        px = cp.ascontiguousarray(pos[:, 0]); py = cp.ascontiguousarray(pos[:, 1])
        cell_id = cp.empty(N, dtype=cp.int32)
        mod.get_function("assign_cells")((grid,),(BLOCK,),(px,py,cell_id,cp.int32(N),cp.int32(n_cells_1d),cs))
        si, cst = self._cell_list_gpu(cell_id, n_cells_1d * n_cells_1d, N)
        cos_t = cp.cos(theta); sin_t = cp.sin(theta)
        ac = cp.empty(N, dtype=cp.float32); asin = cp.empty(N, dtype=cp.float32)
        uf = int(fov_rad < cfg._two_pi - cfg._eps_fov and N > 1)
        ch = cp.float32(cfg.get_cos_half(fov_rad))
        mod.get_function("metric_cell_avg")(
            (grid,),(BLOCK,),
            (px,py,cos_t,sin_t,theta,si,cst,ac,asin,
             cp.int32(N),cp.int32(n_cells_1d),cs,cp.float32(cfg.L),
             cp.float32(cfg._r_sq),cp.int32(uf),ch))
        theta[:] = cp.arctan2(asin, ac) + noise
        return (*self._move(pos, theta, N), )

    # ------------------------------------------------------------------
    # 헬퍼
    # ------------------------------------------------------------------

    def _pbc(self, pos):
        L = self.cfg.L
        dx = pos[:, 0:1] - pos[:, 0]; dy = pos[:, 1:2] - pos[:, 1]
        dx -= L * cp.round(dx / L); dy -= L * cp.round(dy / L)
        return dx, dy

    def _metric_fov(self, ct, st, dx, dy, dsq, fr):
        cfg = self.cfg
        nm = cp.sqrt(cp.maximum(dsq, cfg._eps_sq))
        dot = cp.clip(ct[:, None] * (dx / nm) + st[:, None] * (dy / nm), -1.0, 1.0)
        fov_ok = dot >= cfg.get_cos_half(fr)
        cp.fill_diagonal(fov_ok, False)
        return (dsq < cfg._r_sq) & (dsq > cfg._eps_sq) & fov_ok

    def _fov_filter(self, ct, st, dx, dy, dsq, fr):
        cfg = self.cfg
        nm = cp.sqrt(cp.maximum(dsq, cfg._eps_sq))
        dot = cp.clip(ct[:, None] * (dx / nm) + st[:, None] * (dy / nm), -1.0, 1.0)
        return cp.where(dot >= cfg.get_cos_half(fr), dsq, cp.inf)

    @staticmethod
    def _wavg(th, ct, st, mask, dt):
        n = mask.sum(axis=1).astype(dt)
        sc = cp.where(mask, ct, 0.0).sum(axis=1); ss = cp.where(mask, st, 0.0).sum(axis=1)
        s = cp.where(n > 0, n, 1.0)
        return cp.where(n > 0, cp.arctan2(ss / s, sc / s), th)

    @staticmethod
    def _tavg(th, ct, st, part, valid, dt):
        sc = cp.where(valid, ct[part], 0.0).sum(axis=1)
        ss = cp.where(valid, st[part], 0.0).sum(axis=1)
        n = valid.sum(axis=1).astype(dt); s = cp.where(n > 0, n, 1.0)
        return cp.where(n > 0, cp.arctan2(ss / s, sc / s), th)

    def _move(self, pos, theta, N):
        cn = cp.cos(theta); sn = cp.sin(theta)
        L, vdt = self.cfg.L, self.cfg._vdt
        pos[:, 0] = (pos[:, 0] + vdt * cn) % L
        pos[:, 1] = (pos[:, 1] + vdt * sn) % L
        phi = float(cp.sqrt(cn.sum() ** 2 + sn.sum() ** 2).get()) / N
        return pos, theta, phi

    def _cell_list_gpu(self, cell_id, n_cells, N):
        """GPU 전용 셀 리스트 구축. CPU 왕복 없이 100% GPU에서 처리."""
        mod = self._get_mod()
        si = cp.argsort(cell_id).astype(cp.int32)
        scid = cell_id[si]
        cs = cp.full(n_cells + 1, N, dtype=cp.int32)
        b = cp.where(cp.diff(cp.concatenate([cp.array([-1], dtype=cp.int32), scid])) != 0)[0]
        cs[scid[b]] = b.astype(cp.int32)

        # 빈 셀 역방향 전파: GPU 커널로 처리 (기존 .get() CPU 왕복 제거)
        mod.get_function("backpropagate_cells")(
            (1,), (1,),
            (cs, cp.int32(n_cells), cp.int32(N)),
        )
        return si, cs