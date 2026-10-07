import numpy as np
from metrics import evaluate_scheduled_set


def greedy_user_selection(
    H_eff_all: np.ndarray,
    F_RF: np.ndarray,
    Ns: int,
    Pmax: float,
    sigma2: float,
    stop_if_no_improvement: bool = True,
):
    
    H_eff_all = np.asarray(H_eff_all, dtype=complex)
    if H_eff_all.ndim != 2:
        raise ValueError("H_eff_all must have shape (K, NRF)")

    K, NRF = H_eff_all.shape
    max_users = min(Ns, NRF, K)
    if max_users < 1:
        raise ValueError("At least one user must be selectable")

    # Ref. [30] greedy start: strongest effective channel.
    norms = np.sum(np.abs(H_eff_all) ** 2, axis=1)
    first = int(np.argmax(norms))
    selected = [first]
    remaining = [k for k in range(K) if k != first]

    best_rate, gammas, F_BB, eta, alpha = evaluate_scheduled_set(
        H_eff_all, selected, F_RF, Pmax, sigma2
    )
    best_payload = (best_rate, gammas, F_BB, eta, alpha)

    while len(selected) < max_users and remaining:
        candidates = []
        for u in remaining:
            trial = selected + [u]
            rate, g, F, e, a = evaluate_scheduled_set(
                H_eff_all, trial, F_RF, Pmax, sigma2
            )
            candidates.append((rate, u, g, F, e, a))

        # Deterministic tie break: larger rate, then smaller user index.
        candidates.sort(key=lambda x: (-x[0], x[1]))
        rate, u, g, F, e, a = candidates[0]

        if stop_if_no_improvement and rate <= best_rate + 1e-12:
            break

        selected.append(u)
        remaining.remove(u)
        best_rate = rate
        best_payload = (rate, g, F, e, a)

    rate, gammas, F_BB, eta, alpha = best_payload
    return selected, rate, gammas, F_BB, eta, alpha
