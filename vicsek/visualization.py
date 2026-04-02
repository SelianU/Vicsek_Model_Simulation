"""
Vicsek Model — 실시간 시각화 (GPU 전용, #8)

변경사항: topologic (#11), η [0,1] (#5), GPU 전용 제한 (#8)
"""
from __future__ import annotations

import importlib
import os

import numpy as np

from .config import SimConfig
from .convergence import compute_n_rings


def _select_backend() -> str:
    import matplotlib as _mpl
    for be, mod in [("TkAgg","tkinter"),("Qt5Agg","PyQt5"),("Qt6Agg","PyQt6"),("wxAgg","wx"),("GTK4Agg","gi")]:
        try: importlib.import_module(mod); _mpl.use(be); return be
        except: continue
    _mpl.use("Agg"); return "Agg"

_BACKEND = _select_backend()

import matplotlib, matplotlib.animation as animation, matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class VisualizationRunner:
    """GPU 전용 실시간 시각화 (#8)."""

    ARROW_SCALE = 0.018
    MAX_HIST = 500

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        from .simulator_gpu import VicsekSimulatorGPU
        self.sim = VicsekSimulatorGPU(cfg)

    def run(self, N, fov_deg, eta, model="metric",
            max_frames=2000, interval_ms=30, save_path=None):
        import cupy as cp
        cfg = self.cfg
        fov_rad = float(np.deg2rad(fov_deg))
        chunk = cfg.noise_chunk; L = cfg.L

        def _do_step(pos, theta, noise):
            return self.sim.step_auto(model, pos, theta, noise, N, fov_rad)

        pos = cp.random.uniform(0.0, L, (N, 2), dtype=cfg.dtype)
        theta = cp.random.uniform(-np.pi, np.pi, (N,), dtype=cfg.dtype)

        # 노이즈: η × uniform(-π, π) (#5)
        def _noise_chunk():
            return eta * cp.random.uniform(-np.pi, np.pi, (chunk, N), dtype=cfg.dtype)

        noise_buf = _noise_chunk(); buf_pos = 0
        phi_history = []

        fig = plt.figure(figsize=(12, 5.5), facecolor="#0d0d1a")
        model_label = "METRIC" if model == "metric" else "TOPOLOGIC"
        fig.suptitle(
            f"Vicsek Model  |  {model_label}  N={N}  FOV={fov_deg}°  η={eta:.3f}",
            color="white", fontsize=13, fontweight="bold", y=0.97)
        gs = GridSpec(1, 2, figure=fig, left=0.04, right=0.97, bottom=0.08, top=0.91, wspace=0.28)
        ax_sim = fig.add_subplot(gs[0, 0]); ax_phi = fig.add_subplot(gs[0, 1])
        self._style(ax_sim, ax_phi, L)

        p_np = pos.get(); t_np = theta.get(); al = L * self.ARROW_SCALE
        quiver = ax_sim.quiver(p_np[:,0],p_np[:,1],np.cos(t_np),np.sin(t_np),
            color=self._colors(t_np),scale=1/al,scale_units="xy",width=0.003,headwidth=4,headlength=5,headaxislength=4.5)
        phi_line, = ax_phi.plot([],[],color="#00e5ff",linewidth=1.2,alpha=0.9)
        phi_dot, = ax_phi.plot([],[],"o",color="#ff4081",markersize=5,zorder=5)
        st_txt = ax_sim.text(0.02,0.97,"",transform=ax_sim.transAxes,color="white",fontsize=9,va="top")
        ph_txt = ax_phi.text(0.98,0.97,"",transform=ax_phi.transAxes,color="#ff4081",fontsize=9,va="top",ha="right")

        state = {"pos":pos,"theta":theta,"nb":noise_buf,"bp":buf_pos,"f":0}

        def update(_):
            if state["bp"] >= chunk: state["nb"] = _noise_chunk(); state["bp"] = 0
            noise = state["nb"][state["bp"]]; state["bp"] += 1
            state["pos"],state["theta"],phi = _do_step(state["pos"],state["theta"],noise)
            state["f"] += 1; phi_history.append(phi)
            p = state["pos"].get(); t = state["theta"].get()
            quiver.set_offsets(p); quiver.set_UVC(np.cos(t),np.sin(t)); quiver.set_color(self._colors(t))
            h = phi_history[-self.MAX_HIST:]
            xs = range(max(0,state["f"]-self.MAX_HIST),state["f"])
            phi_line.set_data(list(xs),h); phi_dot.set_data([state["f"]-1],[phi])
            ax_phi.set_xlim(max(0,state["f"]-self.MAX_HIST),max(self.MAX_HIST,state["f"]))
            st_txt.set_text(f"step {state['f']:,}"); ph_txt.set_text(f"φ = {phi:.3f}")
            return quiver,phi_line,phi_dot,st_txt,ph_txt

        frames = max_frames if max_frames > 0 else None
        ani = animation.FuncAnimation(fig,update,frames=frames,interval=interval_ms,blit=True)
        if save_path: self._save(ani,save_path,max_frames,interval_ms)
        else:
            be = matplotlib.get_backend()
            if be.lower() == "agg":
                print("\n[ERROR] 인터랙티브 백엔드 없음. --viz-save 사용 권장.\n")
            else:
                print(f"[INFO] 백엔드: {be}"); plt.show()
        plt.close(fig)

    @staticmethod
    def _colors(t):
        h = (t / (2*np.pi) + 0.5) % 1.0
        return matplotlib.colors.hsv_to_rgb(np.stack([h,np.ones_like(h)*0.9,np.ones_like(h)],axis=-1))

    @staticmethod
    def _style(ax_s, ax_p, L):
        pb="#131326"; gc="#2a2a4a"; lc="#aaaacc"
        for ax in (ax_s,ax_p):
            ax.set_facecolor(pb)
            for sp in ax.spines.values(): sp.set_edgecolor(gc)
            ax.tick_params(colors=lc,labelsize=8); ax.xaxis.label.set_color(lc); ax.yaxis.label.set_color(lc)
        ax_s.set_xlim(0,L);ax_s.set_ylim(0,L);ax_s.set_aspect("equal")
        ax_s.set_xlabel("x");ax_s.set_ylabel("y");ax_s.set_title("Particle Field",color="white",fontsize=10,pad=6)
        ax_s.grid(True,color=gc,linewidth=0.4,alpha=0.5)
        ax_p.set_ylim(-0.05,1.05);ax_p.set_xlabel("time step");ax_p.set_ylabel("order parameter φ")
        ax_p.set_title("Order Parameter φ(t)",color="white",fontsize=10,pad=6)
        ax_p.grid(True,color=gc,linewidth=0.4,alpha=0.5)
        ax_p.axhline(1.0,color="#444466",linewidth=0.7,linestyle="--")
        ax_p.axhline(0.0,color="#444466",linewidth=0.7,linestyle="--")

    @staticmethod
    def _save(ani,path,nf,ims):
        ext = os.path.splitext(path)[1].lower(); fps = max(1,round(1000/ims))
        print(f"\n저장 중: {path} (fps={fps}, frames={nf})")
        w = animation.PillowWriter(fps=fps) if ext==".gif" else animation.FFMpegWriter(fps=fps,bitrate=1800)
        try: ani.save(path,writer=w); print(f"완료: {path}")
        except Exception as e: print(f"[ERROR] 저장 실패: {e}")