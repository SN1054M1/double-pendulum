import numpy as np


class PhysParams:
    def __init__(
        self,
        m1=1.0,
        m2=1.0,
        L1=1.0,
        L2=1.0,
        g=10.0,
        den_epsilon=1e-12,
        omega_clip=200.0,
        accel_clip=5e4,
    ):
        self.m1 = float(m1)
        self.m2 = float(m2)
        self.L1 = float(L1)
        self.L2 = float(L2)
        self.g = float(g)
        self.den_epsilon = float(den_epsilon)
        self.omega_clip = float(omega_clip)
        self.accel_clip = float(accel_clip)


def make_theta_vals(n, tmin, tmax):
    if n <= 0:
        raise ValueError("n must be > 0")
    step = (tmax - tmin) / n
    return (tmin + (np.arange(n) + 0.5) * step).astype(np.float64)


def frame_times(total_time, fps):
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if total_time < 0:
        raise ValueError("total_time must be >= 0")
    n_frames = int(np.floor(total_time * fps)) + 1
    return np.arange(n_frames, dtype=np.float64) / fps


def make_render_targets(total_time, fps, frame_stride=1):
    stride = max(1, int(frame_stride))
    targets = frame_times(total_time, fps)[::stride]
    effective_fps = fps / stride
    return targets, effective_fps


def angles_to_xy(theta1, theta2, p):
    x1 = p.L1 * np.sin(theta1)
    y1 = -p.L1 * np.cos(theta1)
    x2 = x1 + p.L2 * np.sin(theta2)
    y2 = y1 - p.L2 * np.cos(theta2)
    return x1, y1, x2, y2


def angles_to_rgb(theta1, theta2):
    r = 0.5 * (np.cos(theta1) + 1.0)
    gch = 0.5 * (np.cos(theta2) + 1.0)
    b = 0.5 * (np.cos(theta2 - theta1) + 1.0)
    return np.stack((r, gch, b), axis=-1).astype(np.float32)


def _safe_denominator(x, eps):
    sign = np.where(x >= 0.0, 1.0, -1.0)
    return np.where(np.abs(x) < eps, sign * eps, x)


def _finite_clip(x, clip_abs):
    y = np.nan_to_num(x, nan=0.0, posinf=clip_abs, neginf=-clip_abs)
    return np.clip(y, -clip_abs, clip_abs)


def _wrap_angle(theta):
    return np.remainder(theta + np.pi, 2.0 * np.pi) - np.pi


def deriv(theta1, omega1, theta2, omega2, p):
    omega1 = _finite_clip(omega1, p.omega_clip)
    omega2 = _finite_clip(omega2, p.omega_clip)

    d = theta2 - theta1
    sin_d = np.sin(d)
    cos_d = np.cos(d)

    den1 = (p.m1 + p.m2) * p.L1 - p.m2 * p.L1 * cos_d * cos_d
    den2 = (p.L2 / p.L1) * den1
    den1 = _safe_denominator(den1, p.den_epsilon)
    den2 = _safe_denominator(den2, p.den_epsilon)

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        domega1 = (
            p.m2 * p.L1 * omega1 * omega1 * sin_d * cos_d
            + p.m2 * p.g * np.sin(theta2) * cos_d
            + p.m2 * p.L2 * omega2 * omega2 * sin_d
            - (p.m1 + p.m2) * p.g * np.sin(theta1)
        ) / den1

        domega2 = (
            -p.m2 * p.L2 * omega2 * omega2 * sin_d * cos_d
            + (p.m1 + p.m2) * (
                p.g * np.sin(theta1) * cos_d
                - p.L1 * omega1 * omega1 * sin_d
                - p.g * np.sin(theta2)
            )
        ) / den2

    domega1 = _finite_clip(domega1, p.accel_clip)
    domega2 = _finite_clip(domega2, p.accel_clip)

    return omega1, domega1, omega2, domega2


def rk4_step(theta1, omega1, theta2, omega2, dt, p):
    k1 = deriv(theta1, omega1, theta2, omega2, p)
    k2 = deriv(
        theta1 + 0.5 * dt * k1[0],
        omega1 + 0.5 * dt * k1[1],
        theta2 + 0.5 * dt * k1[2],
        omega2 + 0.5 * dt * k1[3],
        p,
    )
    k3 = deriv(
        theta1 + 0.5 * dt * k2[0],
        omega1 + 0.5 * dt * k2[1],
        theta2 + 0.5 * dt * k2[2],
        omega2 + 0.5 * dt * k2[3],
        p,
    )
    k4 = deriv(
        theta1 + dt * k3[0],
        omega1 + dt * k3[1],
        theta2 + dt * k3[2],
        omega2 + dt * k3[3],
        p,
    )

    with np.errstate(over="ignore", invalid="ignore"):
        theta1_next = theta1 + (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        omega1_next = omega1 + (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        theta2_next = theta2 + (dt / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        omega2_next = omega2 + (dt / 6.0) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])

    theta1_next = _wrap_angle(_finite_clip(theta1_next, 1e12))
    theta2_next = _wrap_angle(_finite_clip(theta2_next, 1e12))
    omega1_next = _finite_clip(omega1_next, p.omega_clip)
    omega2_next = _finite_clip(omega2_next, p.omega_clip)

    return theta1_next, omega1_next, theta2_next, omega2_next


def simulate_single_sampled(theta1_0, theta2_0, targets, sim_dt, p):
    if sim_dt <= 0:
        raise ValueError("sim_dt must be > 0")

    targets = np.asarray(targets, dtype=np.float64)
    if targets.ndim != 1:
        raise ValueError("targets must be a 1D array")
    if targets.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if np.any(np.diff(targets) < 0):
        raise ValueError("targets must be non-decreasing")

    t = 0.0
    theta1, omega1 = theta1_0, 0.0
    theta2, omega2 = theta2_0, 0.0

    out = np.empty((targets.size, 4), dtype=np.float32)
    eps = 1e-12

    for k, target in enumerate(targets):
        while t + sim_dt < target - eps:
            theta1, omega1, theta2, omega2 = rk4_step(theta1, omega1, theta2, omega2, sim_dt, p)
            t += sim_dt

        n_theta1, _, n_theta2, _ = rk4_step(theta1, omega1, theta2, omega2, sim_dt, p)

        alpha = (target - t) / sim_dt
        alpha = min(1.0, max(0.0, alpha))

        i_theta1 = theta1 + alpha * (n_theta1 - theta1)
        i_theta2 = theta2 + alpha * (n_theta2 - theta2)

        out[k] = angles_to_xy(i_theta1, i_theta2, p)

    return out


def core_self_check():
    """Run a tiny deterministic simulation to ensure core numerics are finite."""
    p = PhysParams()
    targets = np.array([0.0, 0.02, 0.05, 0.1], dtype=np.float64)
    out = simulate_single_sampled(0.3, -0.2, targets, 0.001, p)
    if out.shape != (targets.size, 4):
        raise RuntimeError(f"Unexpected sampled shape: {out.shape}")
    if not np.all(np.isfinite(out)):
        raise RuntimeError("Non-finite values found in core simulation output")
    return True