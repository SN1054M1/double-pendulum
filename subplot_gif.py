from pathlib import Path
from collections import deque

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
from tqdm import tqdm

from dp_core import (
    PhysParams,
    core_self_check,
    make_theta_vals,
    make_render_targets,
    simulate_single_sampled,
)


# ============================================================
# Paths and output
# ============================================================
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

OUTPUT_GIF = BASE_DIR / "double_pendulum_subplot.gif"

# ============================================================
# Subplot + GIF only (normal version)
# ============================================================
MAKE_GIF_SUBPLOT = True

subplot_n = 3
theta1_min, theta1_max = -np.pi, np.pi
theta2_min, theta2_max = -np.pi, np.pi

sim_dt = 0.01
T = 10.0

gif_fps = 20
gif_dpi = 130
gif_frame_stride = 1

subplot_trail = True
subplot_trail_len = 40
subplot_rod_color = "crimson"
subplot_trail_color = "0.72"

phys = PhysParams(m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=10.0)


def make_subplot_gif():
    core_self_check()

    thetas1 = make_theta_vals(subplot_n, theta1_min, theta1_max)
    thetas2 = make_theta_vals(subplot_n, theta2_min, theta2_max)
    targets, out_fps = make_render_targets(T, gif_fps, gif_frame_stride)

    all_trajs = []
    for th2 in tqdm(
        thetas2,
        desc="Precompute rows",
        unit="row",
        miniters=1,
        mininterval=0.05,
        smoothing=0.15,
        dynamic_ncols=True,
    ):
        row = []
        for th1 in thetas1:
            row.append(simulate_single_sampled(th1, th2, targets, sim_dt, phys))
        all_trajs.append(row)

    fig_size = max(6.0, subplot_n * 1.0)
    fig, axes = plt.subplots(
        subplot_n,
        subplot_n,
        figsize=(fig_size, fig_size),
        sharex=True,
        sharey=True,
    )

    if subplot_n == 1:
        axes = np.array([[axes]])

    for ax in axes.flat:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(-2.15, 2.15)
        ax.set_ylim(-2.15, 2.15)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    rod_lines = []
    trace_lines = []
    traces = []

    for i in range(subplot_n):
        for j in range(subplot_n):
            ax = axes[subplot_n - 1 - i, j]
            if subplot_trail:
                tl, = ax.plot([], [], color=subplot_trail_color, lw=0.7)
                trace_lines.append(tl)
                traces.append(deque(maxlen=max(1, int(subplot_trail_len))))
            else:
                trace_lines.append(None)
                traces.append(None)

            rl, = ax.plot([], [], "-o", color=subplot_rod_color, lw=1.0, ms=2.0)
            rod_lines.append(rl)

    def init():
        for rl in rod_lines:
            rl.set_data([], [])
        if subplot_trail:
            for tl in trace_lines:
                if tl is not None:
                    tl.set_data([], [])
            for tr in traces:
                if tr is not None:
                    tr.clear()
        return rod_lines + [t for t in trace_lines if t is not None]

    def update(frame_idx):
        idx = 0
        for i in range(subplot_n):
            for j in range(subplot_n):
                traj = all_trajs[i][j]
                x1, y1, x2, y2 = traj[frame_idx]

                if subplot_trail:
                    traces[idx].append((x2, y2))
                    tx = [p[0] for p in traces[idx]]
                    ty = [p[1] for p in traces[idx]]
                    trace_lines[idx].set_data(tx, ty)

                rod_lines[idx].set_data([0.0, x1, x2], [0.0, y1, y2])
                idx += 1

        artists = rod_lines[:]
        if subplot_trail:
            artists.extend([t for t in trace_lines if t is not None])
        return artists

    plt.subplots_adjust(wspace=0.0, hspace=0.0, left=0.0, right=1.0, bottom=0.0, top=1.0)

    writer = PillowWriter(fps=out_fps)
    with writer.saving(fig, str(OUTPUT_GIF), dpi=gif_dpi):
        init()
        for fi in tqdm(
            range(targets.size),
            desc="Writing GIF",
            unit="frame",
            miniters=1,
            mininterval=0.03,
            smoothing=0.1,
            dynamic_ncols=True,
        ):
            update(fi)
            writer.grab_frame()

    plt.close(fig)
    print(f"saved: {OUTPUT_GIF}")


if __name__ == "__main__":
    if not MAKE_GIF_SUBPLOT:
        raise ValueError("Set MAKE_GIF_SUBPLOT = True to run this subplot GIF script.")
    make_subplot_gif()
