import numpy as np


def transmit_power(F_RF: np.ndarray, F_BB: np.ndarray) -> float:
    # Eq. (11): ||F_RF F_BB||_F^2
    return float(np.linalg.norm(F_RF @ F_BB, "fro") ** 2)


def rzf_precoder(H_eff: np.ndarray, F_RF: np.ndarray, Pmax: float, sigma2: float):
    # Eq. (20) + Eq. (11): RZF with power normalization
    H_eff = np.asarray(H_eff, dtype=complex)
    F_RF = np.asarray(F_RF, dtype=complex)
    if H_eff.ndim != 2 or F_RF.ndim != 2:
        raise ValueError("H_eff and F_RF must be matrices")

    S, NRF = H_eff.shape
    if S < 1:
        raise ValueError("At least one scheduled user is required")
    if F_RF.shape[1] != NRF:
        raise ValueError("F_RF and H_eff have inconsistent NRF dimensions")
    if Pmax <= 0.0 or sigma2 <= 0.0:
        raise ValueError("Pmax and sigma2 must be positive")

    alpha_rzf = S * sigma2 / Pmax
    A = H_eff @ H_eff.conj().T + alpha_rzf * np.eye(S)

    # [solve(A, H_eff)]^H = H_eff^H A^{-1}
    W = np.linalg.solve(A, H_eff).conj().T
    pre_power = transmit_power(F_RF, W)
    if pre_power <= 0.0:
        raise FloatingPointError("Unnormalized RZF precoder has zero power")

    eta = np.sqrt(Pmax / pre_power)
    F_BB = eta * W
    return F_BB, float(eta), float(alpha_rzf)
