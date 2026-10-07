import numpy as np
from em import pattern_gain_at


def steering_vector(theta: float, Nt: int) -> np.ndarray: # Eq. (12): normalized ULA steering vector a(theta)
    n = np.arange(Nt)
    return np.exp(1j * np.pi * n * np.sin(theta)) / np.sqrt(Nt)


def em_configured_channel_direct(
    path_gains: np.ndarray,
    aods: np.ndarray,
    g: np.ndarray,
    Nt: int,
) -> np.ndarray: # Eq. (14): direct h_k^H F_EM(q), never builds Eq. (6)
    path_gains = np.asarray(path_gains, dtype=complex)
    aods = np.asarray(aods, dtype=float)
    if path_gains.ndim != 2:
        raise ValueError("path_gains must have shape (K, L)")
    K, L = path_gains.shape
    if aods.shape != (K, L):
        raise ValueError("aods must have the same (K, L) shape as path_gains")
    if L < 1:
        raise ValueError("L must be >= 1")

    out = np.zeros((K, Nt), dtype=complex)
    scale = np.sqrt(Nt / L)
    for k in range(K):
        row = np.zeros(Nt, dtype=complex)
        for ell in range(L):
            theta = float(aods[k, ell])
            row += (
                np.conj(path_gains[k, ell])
                * pattern_gain_at(g, theta)
                * steering_vector(theta, Nt).conj()
            )
        out[k] = scale * row
    return out


def effective_channels(H_em: np.ndarray, F_RF: np.ndarray) -> np.ndarray: # Eq. (18)-(19) helper: effective channels after EM + RF layers
    H_em = np.asarray(H_em, dtype=complex)
    F_RF = np.asarray(F_RF, dtype=complex)
    if H_em.ndim != 2 or F_RF.ndim != 2 or H_em.shape[1] != F_RF.shape[0]:
        raise ValueError("Incompatible H_em and F_RF dimensions")
    return H_em @ F_RF
