import numpy as np


def wrap_angle(x):  # Eq. (4) helper: wrap angle to [-pi, pi)
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


def angular_grid(M: int) -> np.ndarray: # Eq. (5): vartheta_i = -pi + 2*pi*(i-1)/M
    return -np.pi + 2.0 * np.pi * np.arange(M) / M


def em_tilt(q: int, N_EM: int, psi_max: float) -> float: # Eq. (3): EM tilt psi_q
    if not 1 <= q <= N_EM:
        raise ValueError(f"q must be in [1, {N_EM}], got {q}")
    return -psi_max + ((2 * q - 1) * psi_max) / N_EM


def em_tilts(N_EM: int, psi_max: float) -> np.ndarray: # Eq. (3): all EM tilts
    return np.array([em_tilt(q, N_EM, psi_max) for q in range(1, N_EM + 1)])


def radiation_pattern_db(vartheta, psi_q: float, phi_3db: float, omega_max_db: float): # Eq. (4): power pattern Omega_q(vartheta) [dB]
    delta = wrap_angle(np.asarray(vartheta) - psi_q)
    return -np.minimum(12.0 * (delta / phi_3db) ** 2, omega_max_db)


def _normalize_amplitude(raw_amp: np.ndarray, M: int):
    energy = float(np.sum(np.abs(raw_amp) ** 2))
    if energy <= 0.0:
        raise FloatingPointError("Pattern energy must be positive")
    G = M / energy
    return np.sqrt(G) * raw_amp, float(G)


def em_amplitude_pattern(
    q: int,
    N_EM: int,
    psi_max: float,
    phi_3db: float,
    omega_max_db: float,
    M: int,
):
    # Eq. (5): sampled, normalized amplitude pattern g_q
    psi_q = em_tilt(q, N_EM, psi_max)
    grid = angular_grid(M)
    omega_db = radiation_pattern_db(grid, psi_q, phi_3db, omega_max_db)
    raw_amp = 10.0 ** (omega_db / 20.0)
    g, Gq = _normalize_amplitude(raw_amp, M)
    return g.astype(float), grid, Gq


def broadside_pattern(phi_3db: float, omega_max_db: float, M: int): # Eq. (4)-(5) with psi_0 = 0: no-EM broadside pattern g_0
    grid = angular_grid(M)
    omega_db = radiation_pattern_db(grid, 0.0, phi_3db, omega_max_db)
    raw_amp = 10.0 ** (omega_db / 20.0)
    g0, G0 = _normalize_amplitude(raw_amp, M)
    return g0.astype(float), grid, G0


def build_em_codebook(config): # Eq. (3)-(5): build {g_1,...,g_NEM}
    patterns = []
    for q in range(1, config.N_EM + 1):
        g, _, _ = em_amplitude_pattern(
            q,
            config.N_EM,
            config.psi_max,
            config.phi_3db,
            config.omega_max_db,
            config.M,
        )
        patterns.append(g)
    return np.stack(patterns, axis=0), em_tilts(config.N_EM, config.psi_max)


def nearest_angular_index(theta: float, M: int) -> int: # Eq. (14) helper: i(theta), defined with Eq. (13)
    grid = angular_grid(M)
    distances = np.abs(wrap_angle(theta - grid))
    return int(np.argmin(distances))


def pattern_gain_at(g: np.ndarray, theta: float) -> float: # Eq. (14): g_q(theta) = [g_q]_{i(theta)}
    return float(g[nearest_angular_index(theta, len(g))])


def normalized_pattern_peak_db(g: np.ndarray) -> float: # S1 acceptance helper
    return float(20.0 * np.log10(np.max(np.abs(g))))
