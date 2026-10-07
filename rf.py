import numpy as np


def quantization_step(beta: int) -> float: # Eq. (8) helper: phase step 2*pi/2^beta
    if beta < 1:
        raise ValueError("beta must be >= 1")
    return 2.0 * np.pi / (2 ** beta)


def quantize_phase(phi, beta: int): # Eq. (8): round phase to nearest beta-bit level
    step = quantization_step(beta)
    phi = np.asarray(phi)
    return step * np.floor(phi / step + 0.5)


def rf_beam(b: int, Nt: int, B: int, beta: int) -> np.ndarray: # Eq. (8): quantized DFT beam f_b
    if not 1 <= b <= B:
        raise ValueError(f"b must be in [1, {B}], got {b}")
    if B < Nt:
        raise ValueError("Eq. (8) assumes B >= Nt")
    nu_b = -1.0 + (2.0 * b - 1.0) / B
    n = np.arange(Nt)
    phase_q = quantize_phase(np.pi * n * nu_b, beta)
    return np.exp(1j * phase_q) / np.sqrt(Nt)


def build_rf_codebook(Nt: int, B: int, beta: int) -> np.ndarray: # Eq. (7)-(8): rows are f_b^T, b=1,...,B
    return np.stack([rf_beam(b, Nt, B, beta) for b in range(1, B + 1)], axis=0)


def rf_precoder(codebook: np.ndarray, beam_indices) -> np.ndarray: # Eq. (9)-(10) helper: assemble distinct beams into F_RF
    codebook = np.asarray(codebook, dtype=complex)
    beam_indices = np.asarray(beam_indices, dtype=int).reshape(-1)
    if codebook.ndim != 2:
        raise ValueError("codebook must have shape (B, Nt)")
    B, _ = codebook.shape
    if len(np.unique(beam_indices)) != len(beam_indices):
        raise ValueError("Eq. (10): RF chains must use distinct beams")
    if np.any(beam_indices < 1) or np.any(beam_indices > B):
        raise ValueError(f"beam indices must lie in [1, {B}]")
    return codebook[beam_indices - 1].T.copy()
