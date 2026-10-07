import numpy as np

from channel import em_configured_channel_direct, effective_channels
from em import build_em_codebook
from metrics import gross_spectral_efficiency, sinr
from precoding import rzf_precoder
from rf import build_rf_codebook, rf_precoder
from scheduling import greedy_user_selection


def run_fixed_configuration(config, path_gains, aods, q: int, beam_indices):
    """Run the fully specified static S1 chain for a fixed (q, b)."""
    em_codebook, _ = build_em_codebook(config)
    rf_codebook = build_rf_codebook(config.Nt, config.B, config.beta)

    if not 1 <= q <= config.N_EM:
        raise ValueError(f"q must be in [1, {config.N_EM}]")

    gq = em_codebook[q - 1]
    F_RF = rf_precoder(rf_codebook, beam_indices)
    H_em_all = em_configured_channel_direct(path_gains, aods, gq, config.Nt)
    H_eff_all = effective_channels(H_em_all, F_RF)

    selected, rate, gammas, F_BB, eta, alpha = greedy_user_selection(
        H_eff_all, F_RF, config.Ns, config.Pmax, config.sigma2
    )

    return {
        "q": int(q),
        "beam_indices": np.asarray(beam_indices, dtype=int),
        "selected_users": selected,
        "H_em_all": H_em_all,
        "H_eff_all": H_eff_all,
        "F_RF": F_RF,
        "F_BB": F_BB,
        "sinr": gammas,
        "gross_se": float(rate),
        "eta": float(eta),
        "alpha_rzf": float(alpha),
    }
