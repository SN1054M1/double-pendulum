# DP Visual

Double pendulum visualization project with one shared RK4 simulation core and two rendering entry scripts:

- Subplot animation to GIF
- Pixel phase-map animation to MP4 (optional multiprocessing)

## Project structure

- `dp_core.py`: shared physics and RK4 integrator.
- `subplot_gif.py`: subplot pendulum animation and GIF export.
- `pixel_mp4_mp.py`: pixel renderer and MP4 export.

## Requirements

Python 3.9+ is recommended.

Install dependencies:

```bash
pip install -r requirements.txt
```

For MP4 output, FFmpeg must be installed and available in your `PATH`.

## Usage

### 1) Subplot GIF

```bash
python subplot_gif.py
```

Output file:

- `double_pendulum_subplot.gif`

### 2) Pixel MP4 (multiprocess)

```bash
python pixel_mp4_mp.py
```

Output file:

- `double_pendulum_pixel.mp4`

## Parameter tuning

Edit parameters at the top of each entry script.

In `subplot_gif.py`:

- `subplot_n`: subplot grid size.
- `sim_dt`: simulation integration step.
- `T`: total simulation time.
- `gif_fps`, `gif_frame_stride`: render cadence.
- Single-process only (teaching-oriented and easier to read).

In `pixel_mp4_mp.py`:

- `pixel_n`: phase map resolution.
- `sim_dt`: simulation integration step.
- `T`: total simulation time.
- `mp4_fps`, `mp4_frame_stride`: render cadence.
- `USE_MULTIPROCESS`, `CPU_USAGE`, `CHUNK_ROWS`: simulation multiprocessing controls.
- `USE_MULTIPROCESS_RENDER`, `RENDER_CPU_USAGE`, `RENDER_CHUNK_ROWS`: render multiprocessing controls.

## Notes

- Simulation always uses `sim_dt` and is not altered by frame skipping.
- Frame skipping only controls render sampling and output FPS.
- Both renderers use the same core equations in `dp_core.py`.
