"""Dynamic channel and mobility model for S2.

Implements manuscript Eqs. (15)-(17), fixed path-angle offsets, the annular
sector drop region, and specular reflection at all four sector boundaries.

Trajectories are generated once and can be cached as compressed NPZ files so
all later schemes can reuse the same common random numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from scipy.special import j0

from config import StaticTHBFConfig
from em import wrap_angle

C_LIGHT = 299_792_458.0


@dataclass(frozen=True)
class DynamicTHBFConfig(StaticTHBFConfig):
    # Table II / Sec. V-A
    fc_hz: float = 28e9
    Ts: float = 0.125e-3
    trajectory_slots: int = 16_000
    r_min_m: float = 30.0
    r_max_m: float = 100.0
    center_dir_min_deg: float = -30.0
    center_dir_max_deg: float = 30.0
    sector_half_width_deg: float = 30.0
    sigma_as_deg: float = 5.0

    @property
    def sector_half_width(self) -> float:
        return float(np.deg2rad(self.sector_half_width_deg))

    @property
    def sigma_as(self) -> float:
        return float(np.deg2rad(self.sigma_as_deg))


def speed_mps(speed_kmh):
    return np.asarray(speed_kmh, dtype=float) / 3.6


def doppler_hz(speed_kmh, fc_hz: float, c: float = C_LIGHT):
    return speed_mps(speed_kmh) * float(fc_hz) / float(c)


def fading_correlation(speed_kmh, fc_hz: float, Ts: float):
    """Eq. (16): rho_k = J0(2*pi*fD,k*Ts)."""
    fD = doppler_hz(speed_kmh, fc_hz)
    return j0(2.0 * np.pi * fD * float(Ts))


def _cross2(a, b) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _inside_sector(point, theta_c, cfg: DynamicTHBFConfig, tol=1e-9) -> bool:
    r = float(np.linalg.norm(point))
    if r < cfg.r_min_m - tol or r > cfg.r_max_m + tol:
        return False
    theta = float(np.arctan2(point[1], point[0]))
    return abs(float(wrap_angle(theta - theta_c))) <= cfg.sector_half_width + tol


def _circle_hit(p, v, radius, tmax, theta_c, cfg, eps=1e-12):
    # ||p+t v||^2 = R^2
    a = float(np.dot(v, v))
    b = 2.0 * float(np.dot(p, v))
    c = float(np.dot(p, p) - radius * radius)
    disc = b * b - 4.0 * a * c
    if disc < 0.0 or a <= 0.0:
        return None
    root = np.sqrt(max(disc, 0.0))
    roots = [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
    best = None
    for t in roots:
        if not (eps < t <= tmax + eps):
            continue
        x = p + t * v
        theta = float(np.arctan2(x[1], x[0]))
        if abs(float(wrap_angle(theta - theta_c))) <= cfg.sector_half_width + 1e-10:
            if best is None or t < best:
                best = float(t)
    return best


def _radial_hit(p, v, boundary_theta, tmax, cfg, eps=1e-12):
    # p + t v = lambda d, lambda in [rmin,rmax]
    d = np.array([np.cos(boundary_theta), np.sin(boundary_theta)], dtype=float)
    den = _cross2(v, d)
    if abs(den) <= 1e-15:
        return None
    t = -_cross2(p, d) / den
    if not (eps < t <= tmax + eps):
        return None
    x = p + t * v
    lam = float(np.dot(x, d))
    if cfg.r_min_m - 1e-10 <= lam <= cfg.r_max_m + 1e-10:
        return float(t)
    return None


def _first_boundary_hit(p, v, tmax, theta_c, cfg):
    candidates = []
    t = _circle_hit(p, v, cfg.r_min_m, tmax, theta_c, cfg)
    if t is not None:
        candidates.append((t, "r_min"))
    t = _circle_hit(p, v, cfg.r_max_m, tmax, theta_c, cfg)
    if t is not None:
        candidates.append((t, "r_max"))

    low = theta_c - cfg.sector_half_width
    high = theta_c + cfg.sector_half_width
    t = _radial_hit(p, v, low, tmax, cfg)
    if t is not None:
        candidates.append((t, "theta_low"))
    t = _radial_hit(p, v, high, tmax, cfg)
    if t is not None:
        candidates.append((t, "theta_high"))

    if not candidates:
        return None
    # Deterministic tie-break in the probability-zero corner case.
    order = {"r_min": 0, "r_max": 1, "theta_low": 2, "theta_high": 3}
    candidates.sort(key=lambda z: (z[0], order[z[1]]))
    return candidates[0]


def _boundary_normal(point, boundary, theta_c, cfg):
    if boundary in ("r_min", "r_max"):
        n = np.asarray(point, dtype=float)
        return n / np.linalg.norm(n)
    theta = theta_c + (-cfg.sector_half_width if boundary == "theta_low" else cfg.sector_half_width)
    d = np.array([np.cos(theta), np.sin(theta)], dtype=float)
    # Either sign is valid for specular reflection.
    return np.array([-d[1], d[0]], dtype=float)


def propagate_with_reflection(position, velocity, dt, theta_c, cfg: DynamicTHBFConfig):
    """Propagate one user for dt with exact specular reflections.

    Returns (new_position, new_velocity, boundaries_hit).
    The speed is preserved by v' = v - 2(v.n)n.
    """
    p = np.asarray(position, dtype=float).copy()
    v = np.asarray(velocity, dtype=float).copy()
    remaining = float(dt)
    hits = []

    for _ in range(16):  # far more than possible at the manuscript slot length
        if remaining <= 1e-15:
            break
        hit = _first_boundary_hit(p, v, remaining, theta_c, cfg)
        if hit is None:
            p = p + remaining * v
            remaining = 0.0
            break
        t_hit, boundary = hit
        p = p + t_hit * v
        remaining -= t_hit
        n = _boundary_normal(p, boundary, theta_c, cfg)
        v = v - 2.0 * float(np.dot(v, n)) * n
        hits.append(boundary)
    else:
        raise RuntimeError("Too many reflections within one slot")

    # Numerical safety only; a substantial violation means reflection logic failed.
    if not _inside_sector(p, theta_c, cfg, tol=2e-8):
        raise FloatingPointError(f"Reflected point left the sector: {p}")
    return p, v, hits


def _complex_normal(rng, shape):
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2.0)


def generate_trajectory(
    cfg: DynamicTHBFConfig,
    seed: int,
    speeds_kmh=30.0,
    slots: int | None = None,
    theta_c: float | None = None,
):
    """Generate one complete dynamic trajectory.

    `speeds_kmh` can be a scalar (main scenario: all users same speed) or an
    array of length K (mixed-mobility scenario).
    """
    T = int(cfg.trajectory_slots if slots is None else slots)
    if T < 2:
        raise ValueError("Trajectory must have at least two slots")
    rng = np.random.default_rng(int(seed))

    if theta_c is None:
        theta_c = float(np.deg2rad(rng.uniform(cfg.center_dir_min_deg, cfg.center_dir_max_deg)))
    else:
        theta_c = float(theta_c)

    speeds = np.asarray(speeds_kmh, dtype=float)
    if speeds.ndim == 0:
        speeds = np.full(cfg.K, float(speeds))
    if speeds.shape != (cfg.K,):
        raise ValueError(f"speeds_kmh must be scalar or shape {(cfg.K,)}")
    speeds_ms = speed_mps(speeds)

    radii0 = rng.uniform(cfg.r_min_m, cfg.r_max_m, size=cfg.K)
    az0 = theta_c + rng.uniform(-cfg.sector_half_width, cfg.sector_half_width, size=cfg.K)
    pos0 = np.column_stack([radii0 * np.cos(az0), radii0 * np.sin(az0)])
    headings0 = rng.uniform(-np.pi, np.pi, size=cfg.K)
    velocities = np.column_stack([speeds_ms * np.cos(headings0), speeds_ms * np.sin(headings0)])

    # Fixed once per trajectory, per manuscript.
    offsets = rng.normal(0.0, cfg.sigma_as, size=(cfg.K, cfg.L))

    positions = np.empty((T, cfg.K, 2), dtype=float)
    headings = np.empty((T, cfg.K), dtype=float)
    reflection_count = np.zeros((T, cfg.K), dtype=np.int16)
    positions[0] = pos0
    headings[0] = headings0

    v_now = velocities.copy()
    for t in range(T - 1):
        for k in range(cfg.K):
            p_new, v_new, hits = propagate_with_reflection(
                positions[t, k], v_now[k], cfg.Ts, theta_c, cfg
            )
            positions[t + 1, k] = p_new
            v_now[k] = v_new
            headings[t + 1, k] = np.arctan2(v_new[1], v_new[0])
            reflection_count[t + 1, k] = len(hits)

    mean_aods = np.arctan2(positions[:, :, 1], positions[:, :, 0])
    aods = wrap_angle(mean_aods[:, :, None] + offsets[None, :, :])

    # Eq. (15), with per-user rho from Eq. (16).
    rho = np.asarray(fading_correlation(speeds, cfg.fc_hz, cfg.Ts), dtype=float)
    alpha = np.empty((T, cfg.K, cfg.L), dtype=np.complex128)
    alpha[0] = _complex_normal(rng, (cfg.K, cfg.L))
    for t in range(T - 1):
        eps = _complex_normal(rng, (cfg.K, cfg.L))
        alpha[t + 1] = rho[:, None] * alpha[t] + np.sqrt(1.0 - rho[:, None] ** 2) * eps

    return {
        "seed": int(seed),
        "theta_c": theta_c,
        "speeds_kmh": speeds,
        "positions": positions,
        "headings": headings,
        "reflection_count": reflection_count,
        "aod_offsets": offsets,
        "mean_aods": mean_aods,
        "aods": np.asarray(aods),
        "alpha": alpha,
        "rho": rho,
    }


def _cache_signature(cfg, seed, speeds_kmh, slots):
    speeds = np.asarray(speeds_kmh, dtype=float)
    if speeds.ndim == 0:
        speeds = np.full(cfg.K, float(speeds))
    payload = {
        "seed": int(seed),
        "speeds_kmh": speeds.tolist(),
        "slots": int(slots),
        "K": cfg.K,
        "L": cfg.L,
        "Ts": cfg.Ts,
        "fc_hz": cfg.fc_hz,
        "r_min_m": cfg.r_min_m,
        "r_max_m": cfg.r_max_m,
        "sector_half_width_deg": cfg.sector_half_width_deg,
        "sigma_as_deg": cfg.sigma_as_deg,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def load_or_generate_trajectory(
    cfg: DynamicTHBFConfig,
    seed: int,
    speeds_kmh=30.0,
    slots: int | None = None,
    cache_dir="data/trajectories",
):
    """Generate once, store with seed/config signature, then reuse bit-for-bit."""
    T = int(cfg.trajectory_slots if slots is None else slots)
    speeds = np.asarray(speeds_kmh, dtype=float)
    if speeds.ndim == 0:
        label = f"v{float(speeds):g}"
    else:
        label = "mixed_" + "-".join(f"{x:g}" for x in speeds)
    path = Path(cache_dir) / f"trajectory_seed{int(seed)}_{label}_T{T}.npz"
    sig = _cache_signature(cfg, seed, speeds_kmh, T)

    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            stored_sig = str(z["signature"].item())
            if stored_sig != sig:
                raise ValueError(f"Cached trajectory signature mismatch: {path}")
            return {key: z[key] for key in z.files if key != "signature"}, path, True

    traj = generate_trajectory(cfg, seed, speeds_kmh=speeds_kmh, slots=T)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, signature=np.array(sig), **traj)
    return traj, path, False


def empirical_lag_correlation(alpha: np.ndarray, max_lag: int):
    """Estimate E[alpha(t+lag) alpha*(t)] / E[|alpha(t)|^2]."""
    x = np.asarray(alpha, dtype=np.complex128)
    if x.ndim < 1:
        raise ValueError("alpha must have a time axis")
    T = x.shape[0]
    out = np.empty(max_lag + 1, dtype=float)
    out[0] = 1.0
    for lag in range(1, max_lag + 1):
        x0 = x[:-lag].reshape(-1)
        x1 = x[lag:].reshape(-1)
        den = np.mean(np.abs(x0) ** 2)
        out[lag] = float(np.real(np.mean(x1 * np.conj(x0)) / den))
    return out
