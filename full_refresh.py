"""Deterministic model-based EM/RF reference for S1.

Implements manuscript Eqs. (34)-(37):
  Gamma_k(q,b): predicted average beam gain with fast fading averaged out.
  Phi(b;q): deterministic distinct-user assignment proxy.
  b_full(q): deterministic greedy full RF refresh.
  Psi(q) = Phi(b_full(q);q), q* = argmax_q Psi(q).

The selector depends on RF-layer angular information (AoDs) only. Instantaneous
path gains alpha_{k,l} are deliberately NOT inputs to q*/RF selection.
"""
from __future__ import annotations

import numpy as np

from channel import steering_vector
from config import StaticTHBFConfig
from em import build_em_codebook, pattern_gain_at
from rf import build_rf_codebook


def _validate_beam_config(beam_indices, B: int, max_len: int) -> np.ndarray:
    b = np.asarray(beam_indices, dtype=int).reshape(-1)
    if len(b) > max_len:
        raise ValueError(f"At most {max_len} RF beams are allowed")
    if len(np.unique(b)) != len(b):
        raise ValueError("RF chains must use distinct beams")
    if np.any(b < 1) or np.any(b > B):
        raise ValueError(f"Beam indices must lie in [1, {B}]")
    return b


def average_beam_gains(cfg: StaticTHBFConfig, aods: np.ndarray, q: int) -> np.ndarray:
    """Eq. (34): Gamma[k,b] for fixed/latest RF-layer AoDs.

    Returns shape (K,B). No instantaneous fast-fading gains appear here.
    """
    aods = np.asarray(aods, dtype=float)
    if aods.shape != (cfg.K, cfg.L):
        raise ValueError(f"aods must have shape {(cfg.K, cfg.L)}, got {aods.shape}")
    if not 1 <= q <= cfg.N_EM:
        raise ValueError(f"q must be in [1, {cfg.N_EM}]")

    gq = build_em_codebook(cfg)[0][q - 1]
    rf = build_rf_codebook(cfg.Nt, cfg.B, cfg.beta)  # (B,Nt), rows f_b^T
    gamma = np.zeros((cfg.K, cfg.B), dtype=float)

    for k in range(cfg.K):
        for ell in range(cfg.L):
            th = float(aods[k, ell])
            g2 = pattern_gain_at(gq, th) ** 2
            a = steering_vector(th, cfg.Nt)
            # |a^H f_b|^2 for all codebook beams.
            proj2 = np.abs(rf @ np.conj(a)) ** 2
            gamma[k] += (cfg.Nt / cfg.L) * g2 * proj2
    return gamma


def _max_distinct_user_assignment(values: np.ndarray) -> float:
    """Maximum sum assigning a distinct user to each column (beam).

    values has shape (K,R), R<=NRF. Dynamic programming gives the exact
    assignment score and deterministic numerical behavior without SciPy.
    """
    K, R = values.shape
    if R == 0:
        return 0.0
    if R > K:
        raise ValueError("Cannot assign more beams than distinct users")

    # mask -> best score after assigning processed beams.
    dp = {0: 0.0}
    for r in range(R):
        nxt = {}
        for mask, score in dp.items():
            for k in range(K):
                if mask & (1 << k):
                    continue
                m2 = mask | (1 << k)
                s2 = score + float(values[k, r])
                if m2 not in nxt or s2 > nxt[m2]:
                    nxt[m2] = s2
        dp = nxt
    return float(max(dp.values()))


def phi_assignment_proxy(cfg: StaticTHBFConfig, gamma: np.ndarray, beam_indices) -> float:
    """Eq. (35), extended naturally to a greedy partial beam prefix.

    For a complete NRF-beam configuration this is exactly Eq. (35). During
    greedy construction the same distinct-user assignment objective is applied
    to the current prefix; the Eq. (35) equal-power factor Pmax/(Ns*sigma2)
    remains unchanged.
    """
    b = _validate_beam_config(beam_indices, cfg.B, cfg.NRF)
    if len(b) == 0:
        return 0.0
    gamma = np.asarray(gamma, dtype=float)
    if gamma.shape != (cfg.K, cfg.B):
        raise ValueError(f"gamma must have shape {(cfg.K, cfg.B)}")
    rho_stream = cfg.Pmax / (cfg.Ns * cfg.sigma2)
    values = np.log2(1.0 + rho_stream * gamma[:, b - 1])
    return _max_distinct_user_assignment(values)


def greedy_rf_full_refresh(cfg: StaticTHBFConfig, aods: np.ndarray, q: int):
    """Deterministic full RF refresh using Eqs. (34)-(35).

    All NRF chains are re-optimized from scratch. At each greedy step, add the
    unused beam giving the largest Phi increase/current-prefix Phi. Ties use
    the smallest beam index, as required by the manuscript.
    """
    gamma = average_beam_gains(cfg, aods, q)
    selected: list[int] = []
    prefix_scores: list[float] = []

    for _ in range(cfg.NRF):
        best_beam = None
        best_score = -np.inf
        for beam in range(1, cfg.B + 1):
            if beam in selected:
                continue
            score = phi_assignment_proxy(cfg, gamma, selected + [beam])
            # Ascending beam loop + strict comparison => smallest-index tie.
            if score > best_score:
                best_score = score
                best_beam = beam
        selected.append(int(best_beam))
        prefix_scores.append(float(best_score))

    beams = np.asarray(selected, dtype=int)
    return beams, float(phi_assignment_proxy(cfg, gamma, beams)), prefix_scores


def select_q_star(cfg: StaticTHBFConfig, aods: np.ndarray):
    """Eqs. (36)-(37): select q* using AoD-based average gains only."""
    scores = np.empty(cfg.N_EM, dtype=float)
    configs = {}
    for q in range(1, cfg.N_EM + 1):
        b_full, score, _ = greedy_rf_full_refresh(cfg, aods, q)
        configs[q] = b_full
        scores[q - 1] = score

    # np.argmax returns first maximum => smallest-q deterministic tie break.
    q_star = int(np.argmax(scores)) + 1
    return q_star, configs[q_star].copy(), scores, configs


def random_distinct_configuration(rng: np.random.Generator, B: int, NRF: int):
    return np.asarray(rng.choice(B, size=NRF, replace=False) + 1, dtype=int)
