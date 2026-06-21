import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

from dp_core import (
    PhysParams,
    angles_to_rgb,
    core_self_check,
    make_render_targets,
    make_theta_vals,
    rk4_step,
)


# ============================================================
# Paths and output
# ============================================================
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

OUTPUT_MP4 = BASE_DIR / "double_pendulum_pixel.mp4"

# ============================================================
# Pixel + MP4 only (multi-process version)
# ============================================================
MAKE_MP4 = True

pixel_n = 99
theta1_min, theta1_max = -np.pi, np.pi
theta2_min, theta2_max = -np.pi, np.pi

sim_dt = 0.01
T = 10.0

mp4_fps = 30
mp4_dpi = 170
mp4_frame_stride = 1

pixel_use_trail = False
pixel_trail_alpha = 0.88

USE_MULTIPROCESS = True
CPU_USAGE = 0.9
CHUNK_ROWS = 32

USE_MULTIPROCESS_RENDER = True
RENDER_CPU_USAGE = CPU_USAGE
RENDER_CHUNK_ROWS = 64

TQDM_PROGRESS_KW = dict(
    miniters=1,
    mininterval=0.03,
    smoothing=0.1,
    dynamic_ncols=True,
)

phys = PhysParams(m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=10.0)


def chunk_ranges(n_rows, chunk_rows):
    out = []
    s = 0
    while s < n_rows:
        e = min(n_rows, s + chunk_rows)
        out.append((s, e))
        s = e
    return out


def workers_by_cpu_usage(usage):
    cpu_total = max(1, os.cpu_count() or 1)
    usage = float(usage)
    usage = min(1.0, max(0.0, usage))
    if usage <= 0.0:
        return 1
    return max(1, min(cpu_total, int(np.ceil(cpu_total * usage))))


def simulate_chunk(args):
    i0, i1, theta1_rows, theta2_rows, omega1_rows, omega2_rows, dt, params = args
    n_theta1, n_omega1, n_theta2, n_omega2 = rk4_step(
        theta1_rows, omega1_rows, theta2_rows, omega2_rows, dt, params
    )
    return i0, i1, n_theta1, n_omega1, n_theta2, n_omega2


def rgb_chunk(args):
    i0, i1, theta1_rows, theta2_rows = args
    return i0, i1, angles_to_rgb(theta1_rows, theta2_rows)


def step_state_single_process(theta1, omega1, theta2, omega2, dt, params):
    return rk4_step(theta1, omega1, theta2, omega2, dt, params)


def step_state_multi_process(theta1, omega1, theta2, omega2, dt, params, ranges, pool, workers):
    jobs = []
    for i0, i1 in ranges:
        jobs.append((
            i0,
            i1,
            theta1[i0:i1],
            theta2[i0:i1],
            omega1[i0:i1],
            omega2[i0:i1],
            dt,
            params,
        ))

    map_chunk = max(1, len(jobs) // max(1, workers * 4))
    for i0, i1, n_th1, n_om1, n_th2, n_om2 in pool.map(simulate_chunk, jobs, chunksize=map_chunk):
        theta1[i0:i1] = n_th1
        omega1[i0:i1] = n_om1
        theta2[i0:i1] = n_th2
        omega2[i0:i1] = n_om2

    return theta1, omega1, theta2, omega2


def rgb_parallel(theta1, theta2, ranges, pool, workers):
    rgb = np.empty((theta1.shape[0], theta1.shape[1], 3), dtype=np.float32)

    jobs = []
    for i0, i1 in ranges:
        jobs.append((i0, i1, theta1[i0:i1], theta2[i0:i1]))

    map_chunk = max(1, len(jobs) // max(1, workers * 4))
    for i0, i1, rgb_rows in pool.map(rgb_chunk, jobs, chunksize=map_chunk):
        rgb[i0:i1] = rgb_rows

    return rgb


def make_pixel_mp4():
    core_self_check()

    if sim_dt <= 0:
        raise ValueError("sim_dt must be > 0")

    thetas1 = make_theta_vals(pixel_n, theta1_min, theta1_max)
    thetas2 = make_theta_vals(pixel_n, theta2_min, theta2_max)
    targets, out_fps = make_render_targets(T, mp4_fps, mp4_frame_stride)

    theta1, theta2 = np.meshgrid(thetas1, thetas2, indexing="xy")
    omega1 = np.zeros_like(theta1, dtype=np.float64)
    omega2 = np.zeros_like(theta2, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7.0, 7.0))
    ax.set_axis_off()
    ax.set_aspect("equal", adjustable="box")

    rgb0 = angles_to_rgb(theta1, theta2)
    im = ax.imshow(
        rgb0,
        origin="lower",
        interpolation="nearest",
        extent=[theta1_min, theta1_max, theta2_min, theta2_max],
    )
    trail_rgb = rgb0.copy() if pixel_use_trail else None

    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)

    writer = FFMpegWriter(fps=out_fps)

    t = 0.0
    fi = 0
    eps = 1e-12
    sim_steps = int(np.ceil(T / sim_dt))
    ranges = chunk_ranges(theta1.shape[0], CHUNK_ROWS)
    render_ranges = chunk_ranges(theta1.shape[0], RENDER_CHUNK_ROWS)

    sim_workers = workers_by_cpu_usage(CPU_USAGE)
    render_workers = workers_by_cpu_usage(RENDER_CPU_USAGE)

    use_mp_sim = USE_MULTIPROCESS and sim_workers > 1
    use_mp_render = USE_MULTIPROCESS_RENDER and render_workers > 1

    with writer.saving(fig, str(OUTPUT_MP4), dpi=mp4_dpi):
        if fi < targets.size and abs(targets[fi]) <= eps:
            im.set_data(trail_rgb if pixel_use_trail else rgb0)
            writer.grab_frame()
            fi += 1

        if use_mp_sim:
            with ProcessPoolExecutor(max_workers=sim_workers) as sim_pool:
                for _ in tqdm(
                    range(sim_steps),
                    desc="Simulating+Writing MP4",
                    unit="step",
                    **TQDM_PROGRESS_KW,
                ):
                    prev_theta1 = theta1.copy()
                    prev_theta2 = theta2.copy()

                    theta1, omega1, theta2, omega2 = step_state_multi_process(
                        theta1,
                        omega1,
                        theta2,
                        omega2,
                        sim_dt,
                        phys,
                        ranges,
                        sim_pool,
                        sim_workers,
                    )

                    t_next = t + sim_dt

                    while fi < targets.size and targets[fi] <= t_next + eps:
                        alpha = (targets[fi] - t) / sim_dt
                        alpha = min(1.0, max(0.0, alpha))

                        i_theta1 = prev_theta1 + alpha * (theta1 - prev_theta1)
                        i_theta2 = prev_theta2 + alpha * (theta2 - prev_theta2)

                        if use_mp_render:
                            rgb = rgb_parallel(i_theta1, i_theta2, render_ranges, sim_pool, sim_workers)
                        else:
                            rgb = angles_to_rgb(i_theta1, i_theta2)

                        if pixel_use_trail:
                            trail_rgb = pixel_trail_alpha * trail_rgb + (1.0 - pixel_trail_alpha) * rgb
                            im.set_data(trail_rgb)
                        else:
                            im.set_data(rgb)

                        writer.grab_frame()
                        fi += 1

                    t = t_next
                    if fi >= targets.size:
                        break
        elif use_mp_render:
            with ProcessPoolExecutor(max_workers=render_workers) as render_pool:
                for _ in tqdm(
                    range(sim_steps),
                    desc="Simulating+Writing MP4",
                    unit="step",
                    **TQDM_PROGRESS_KW,
                ):
                    prev_theta1 = theta1.copy()
                    prev_theta2 = theta2.copy()

                    theta1, omega1, theta2, omega2 = step_state_single_process(
                        theta1,
                        omega1,
                        theta2,
                        omega2,
                        sim_dt,
                        phys,
                    )

                    t_next = t + sim_dt

                    while fi < targets.size and targets[fi] <= t_next + eps:
                        alpha = (targets[fi] - t) / sim_dt
                        alpha = min(1.0, max(0.0, alpha))

                        i_theta1 = prev_theta1 + alpha * (theta1 - prev_theta1)
                        i_theta2 = prev_theta2 + alpha * (theta2 - prev_theta2)

                        rgb = rgb_parallel(i_theta1, i_theta2, render_ranges, render_pool, render_workers)

                        if pixel_use_trail:
                            trail_rgb = pixel_trail_alpha * trail_rgb + (1.0 - pixel_trail_alpha) * rgb
                            im.set_data(trail_rgb)
                        else:
                            im.set_data(rgb)

                        writer.grab_frame()
                        fi += 1

                    t = t_next
                    if fi >= targets.size:
                        break
        else:
            for _ in tqdm(
                range(sim_steps),
                desc="Simulating+Writing MP4",
                unit="step",
                **TQDM_PROGRESS_KW,
            ):
                prev_theta1 = theta1.copy()
                prev_theta2 = theta2.copy()

                theta1, omega1, theta2, omega2 = step_state_single_process(
                    theta1,
                    omega1,
                    theta2,
                    omega2,
                    sim_dt,
                    phys,
                )

                t_next = t + sim_dt

                while fi < targets.size and targets[fi] <= t_next + eps:
                    alpha = (targets[fi] - t) / sim_dt
                    alpha = min(1.0, max(0.0, alpha))

                    i_theta1 = prev_theta1 + alpha * (theta1 - prev_theta1)
                    i_theta2 = prev_theta2 + alpha * (theta2 - prev_theta2)

                    rgb = angles_to_rgb(i_theta1, i_theta2)

                    if pixel_use_trail:
                        trail_rgb = pixel_trail_alpha * trail_rgb + (1.0 - pixel_trail_alpha) * rgb
                        im.set_data(trail_rgb)
                    else:
                        im.set_data(rgb)

                    writer.grab_frame()
                    fi += 1

                t = t_next
                if fi >= targets.size:
                    break

    plt.close(fig)
    print(f"saved: {OUTPUT_MP4}")


if __name__ == "__main__":
    if not MAKE_MP4:
        raise ValueError("Set MAKE_MP4 = True to run this pixel MP4 script.")
    make_pixel_mp4()
