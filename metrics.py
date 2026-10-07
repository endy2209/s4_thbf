import numpy as np


def sinr(H_eff: np.ndarray, F_BB: np.ndarray, sigma2: float) -> np.ndarray: # Eq. (22): user SINR
    H_eff = np.asarray(H_eff, dtype=complex)
    F_BB = np.asarray(F_BB, dtype=complex)
    if H_eff.shape[1] != F_BB.shape[0]:
        raise ValueError("Incompatible H_eff and F_BB dimensions")
    coupling = H_eff @ F_BB
    powers = np.abs(coupling) ** 2
    desired = np.diag(powers)
    interference = np.sum(powers, axis=1) - desired
    return desired / (interference + sigma2)


def gross_spectral_efficiency(sinr_values: np.ndarray) -> float: # Eq. (23): gross SE = sum log2(1 + gamma_k)
    values = np.asarray(sinr_values, dtype=float)
    if np.any(values < 0.0):
        raise ValueError("SINR values must be nonnegative")
    return float(np.sum(np.log2(1.0 + values)))


def evaluate_scheduled_set(H_eff_all, user_indices, F_RF, Pmax, sigma2): # S1 helper: Eq. (20) -> Eq. (22) -> Eq. (23)
    from precoding import rzf_precoder

    user_indices = list(user_indices)
    if not user_indices:
        raise ValueError("user_indices must not be empty")
    H = np.asarray(H_eff_all, dtype=complex)[user_indices]
    F_BB, eta, alpha = rzf_precoder(H, F_RF, Pmax, sigma2)
    gammas = sinr(H, F_BB, sigma2)
    rate = gross_spectral_efficiency(gammas)
    return rate, gammas, F_BB, eta, alpha
