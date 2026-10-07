import numpy as np
from dataclasses import replace

from channel import em_configured_channel_direct, steering_vector
from config import StaticTHBFConfig
from em import (
    broadside_pattern,
    build_em_codebook,
    normalized_pattern_peak_db,
    pattern_gain_at,
)
from full_refresh import (
    average_beam_gains,
    greedy_rf_full_refresh,
    random_distinct_configuration,
    select_q_star,
)
from static_model import run_fixed_configuration
from metrics import gross_spectral_efficiency, sinr
from precoding import rzf_precoder, transmit_power
from rf import build_rf_codebook, rf_precoder


def check_norms_and_phases(cfg):
    em_codebook, _ = build_em_codebook(cfg)
    rf_codebook = build_rf_codebook(cfg.Nt, cfg.B, cfg.beta)

    rf_norm_err = float(np.max(np.abs(np.linalg.norm(rf_codebook, axis=1) - 1.0)))
    steering_err = abs(np.linalg.norm(steering_vector(0.37, cfg.Nt)) - 1.0)
    em_energy_err = float(np.max(np.abs(np.sum(em_codebook ** 2, axis=1) - cfg.M)))

    step = 2.0 * np.pi / (2 ** cfg.beta)
    phases = np.angle(rf_codebook * np.sqrt(cfg.Nt))
    phase_grid_err = float(np.max(np.abs(phases / step - np.round(phases / step))))

    return {
        "rf_norm_max_abs_error": rf_norm_err,
        "steering_norm_abs_error": float(steering_err),
        "em_energy_max_abs_error": em_energy_err,
        "phase_grid_max_error": phase_grid_err,
        "allowed_phase_levels": 2 ** cfg.beta,
    }


def check_power_normalization(cfg):
    rng = np.random.default_rng(1)
    H_eff = (rng.normal(size=(cfg.Ns, cfg.NRF)) + 1j * rng.normal(size=(cfg.Ns, cfg.NRF))) / np.sqrt(2)
    codebook = build_rf_codebook(cfg.Nt, cfg.B, cfg.beta)
    F_RF = rf_precoder(codebook, np.arange(1, cfg.NRF + 1))
    F_BB, _, _ = rzf_precoder(H_eff, F_RF, cfg.Pmax, cfg.sigma2)
    p = transmit_power(F_RF, F_BB)
    rel = abs(p - cfg.Pmax) / cfg.Pmax
    return {"power": p, "Pmax": cfg.Pmax, "relative_error": float(rel)}


def check_eq14_statistics(cfg, draws=100_000, seed=2):
    rng = np.random.default_rng(seed)
    q = 1
    g = build_em_codebook(cfg)[0][q - 1]
    aods = np.deg2rad(np.array([[-20.0, 3.0, 17.0]]))

    gains = (rng.normal(size=(draws, cfg.L)) + 1j * rng.normal(size=(draws, cfg.L))) / np.sqrt(2.0)
    A = np.stack([steering_vector(float(th), cfg.Nt).conj() for th in aods[0]], axis=0)
    gv = np.array([pattern_gain_at(g, float(th)) for th in aods[0]])
    H = np.sqrt(cfg.Nt / cfg.L) * ((np.conj(gains) * gv[None, :]) @ A)
    empirical = float(np.mean(np.sum(np.abs(H) ** 2, axis=1)))
    theory = float((cfg.Nt / cfg.L) * np.sum(gv ** 2))
    rel = abs(empirical - theory) / theory
    return {"empirical": empirical, "theory": theory, "relative_error": float(rel)}


def check_pattern_peak(cfg):
    g, _, _ = broadside_pattern(cfg.phi_3db, cfg.omega_max_db, cfg.M)
    peak = normalized_pattern_peak_db(g)
    return {"peak_db": peak}


def check_no_em_reproducibility(cfg, seed=3):
    def one_run():
        rng = np.random.default_rng(seed)
        gains = (rng.normal(size=(cfg.K, cfg.L)) + 1j * rng.normal(size=(cfg.K, cfg.L))) / np.sqrt(2.0)
        aods = rng.uniform(-np.pi / 3, np.pi / 3, size=(cfg.K, cfg.L))
        g0, _, _ = broadside_pattern(cfg.phi_3db, cfg.omega_max_db, cfg.M)
        H = em_configured_channel_direct(gains, aods, g0, cfg.Nt)
        return gains, aods, H

    a = one_run()
    b = one_run()
    return {"bitwise_equal": all(np.array_equal(x, y) for x, y in zip(a, b))}


def check_q_star_full_refresh(
    cfg,
    gain_draws=4,
    random_configs=16,
    seed=1234,
):
    """S1 acceptance using manuscript Eqs. (34)-(37).

    1) Generate one fixed AoD set.
    2) Select q* and the deterministic full-refresh RF configuration ONCE
       from average beam gains Gamma (fast fading averaged out).
    3) Freeze that selected EM/RF configuration.
    4) Generate a fixed bank of random EM-RF configurations (random q AND
       random distinct RF beams).
    5) Evaluate both the selected configuration and every random EM-RF
       configuration over the SAME independent fast-fading realizations.
    """
    rng = np.random.default_rng(seed)
    aods = rng.uniform(-np.pi / 3, np.pi / 3, size=(cfg.K, cfg.L))

    # Model-based selection is deterministic for these fixed AoDs.
    q_star, b_star, psi_scores, _ = select_q_star(cfg, aods)

    # Random comparison bank: randomize BOTH EM state and RF configuration.
    # Generate once, then reuse the exact same bank over every fading draw.
    random_bank = []
    for _ in range(random_configs):
        q_random = int(rng.integers(1, cfg.N_EM + 1))
        b_random = random_distinct_configuration(rng, cfg.B, cfg.NRF)
        random_bank.append((q_random, b_random))

    # Generate the fading bank once so selected and random configurations are
    # evaluated on exactly the same alpha realizations.
    fading_bank = []
    for _ in range(gain_draws):
        gains = (
            rng.normal(size=(cfg.K, cfg.L))
            + 1j * rng.normal(size=(cfg.K, cfg.L))
        ) / np.sqrt(2.0)
        fading_bank.append(gains)

    selected_rates = []
    random_rates = []
    for gains in fading_bank:
        selected_rates.append(
            run_fixed_configuration(cfg, gains, aods, q_star, b_star)["gross_se"]
        )
        for q_random, b_random in random_bank:
            random_rates.append(
                run_fixed_configuration(
                    cfg, gains, aods, q_random, b_random
                )["gross_se"]
            )

    # Explicit invariance check: selection cannot change when alpha changes,
    # because alpha is not an input to select_q_star.
    q_repeat, b_repeat, _, _ = select_q_star(cfg, aods)
    selection_invariant = bool(
        q_repeat == q_star and np.array_equal(b_repeat, b_star)
    )

    selected_mean = float(np.mean(selected_rates))
    random_mean = float(np.mean(random_rates))
    margin = selected_mean - random_mean
    return {
        "q_star": int(q_star),
        "selected_beams": b_star.tolist(),
        "psi_scores": psi_scores.tolist(),
        "selection_invariant_to_fast_fading": selection_invariant,
        "randomizes_em_and_rf": True,
        "same_fading_draws_for_selected_and_random": True,
        "selected_mean_gross_se": selected_mean,
        "random_mean_gross_se": random_mean,
        "margin_bit_s_hz": float(margin),
        "passed": bool(selection_invariant and selected_mean > random_mean),
        "gain_draws": int(gain_draws),
        "random_em_rf_configurations": int(random_configs),
    }


def _snr_test_configuration_bank(cfg, aods, rng, random_rf_per_q=3):
    """Build a deterministic batch of fixed EM-RF configurations.

    For every EM state q, test its deterministic full-refresh configuration
    plus ``random_rf_per_q`` seeded random distinct RF configurations. This
    covers all EM states and multiple RF configurations without attempting the
    impractical exhaustive ~1e8 configuration space.
    """
    configs = []
    for q in range(1, cfg.N_EM + 1):
        b_full, _, _ = greedy_rf_full_refresh(cfg, aods, q)
        configs.append((int(q), b_full.copy(), "full_refresh"))
        for _ in range(random_rf_per_q):
            b_random = random_distinct_configuration(rng, cfg.B, cfg.NRF)
            configs.append((int(q), b_random, "random_rf"))
    return configs


def check_gross_se_vs_snr(
    cfg,
    snr_db_values=(-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0),
    random_rf_per_q=3,
    seed=2026,
):
    """S1 sanity check: every tested fixed EM-RF configuration increases in SE.

    A single static channel realization (path gains and AoDs) is held fixed
    over the complete SNR sweep. The test covers every EM state and, for each
    q, one deterministic full-refresh RF configuration plus several seeded
    random RF configurations. Each (q,b) pair is frozen over all SNR points;
    only Pmax/sigma^2 changes.
    """
    rng = np.random.default_rng(seed)
    gains = (
        rng.normal(size=(cfg.K, cfg.L))
        + 1j * rng.normal(size=(cfg.K, cfg.L))
    ) / np.sqrt(2.0)
    aods = rng.uniform(-np.pi / 3, np.pi / 3, size=(cfg.K, cfg.L))

    config_bank = _snr_test_configuration_bank(
        cfg, aods, rng, random_rf_per_q=random_rf_per_q
    )

    all_results = []
    global_min_increment = np.inf
    worst = None
    all_increasing = True

    for q_fixed, b_fixed, source in config_bank:
        rates = []
        for snr_db in snr_db_values:
            cfg_snr = replace(cfg, snr_db=float(snr_db))
            rate = run_fixed_configuration(
                cfg_snr, gains, aods, q_fixed, b_fixed
            )["gross_se"]
            rates.append(float(rate))

        increments = np.diff(rates)
        passed_this = bool(np.all(increments > 0.0))
        all_increasing = all_increasing and passed_this
        min_inc = float(np.min(increments))
        if min_inc < global_min_increment:
            global_min_increment = min_inc
            worst = {
                "q": int(q_fixed),
                "beams": b_fixed.tolist(),
                "source": source,
                "gross_se": rates,
                "increments": increments.tolist(),
                "strictly_increasing": passed_this,
            }
        all_results.append((q_fixed, b_fixed, source, rates, increments, passed_this))

    # Keep one representative curve in the log for readability.
    q_rep, b_rep, src_rep, rates_rep, inc_rep, pass_rep = all_results[0]

    return {
        "snr_db": [float(x) for x in snr_db_values],
        "configurations_tested": len(config_bank),
        "em_states_covered": cfg.N_EM,
        "rf_configs_per_em_state": int(1 + random_rf_per_q),
        "fixed_channel_and_aods": True,
        "representative_q": int(q_rep),
        "representative_beams": b_rep.tolist(),
        "representative_source": src_rep,
        "representative_gross_se": rates_rep,
        "representative_increments": inc_rep.tolist(),
        "minimum_increment_over_all_configs": float(global_min_increment),
        "worst_case_configuration": worst,
        "all_configs_strictly_increasing": bool(all_increasing),
        "passed": bool(all_increasing),
    }


def main():
    cfg = StaticTHBFConfig()
    results = {
        "norms_and_phases": check_norms_and_phases(cfg),
        "power_normalization": check_power_normalization(cfg),
        "eq14_statistics": check_eq14_statistics(cfg),
        "pattern_peak": check_pattern_peak(cfg),
        "q_star_full_refresh": check_q_star_full_refresh(cfg),
        "gross_se_vs_snr": check_gross_se_vs_snr(cfg),
        "no_em_reproducibility": check_no_em_reproducibility(cfg),
    }

    for name, value in results.items():
        print(f"[{name}]")
        print(value)
        print()

    assert results["norms_and_phases"]["rf_norm_max_abs_error"] < 1e-12
    assert results["norms_and_phases"]["steering_norm_abs_error"] < 1e-12
    assert results["norms_and_phases"]["em_energy_max_abs_error"] < 1e-9
    assert results["norms_and_phases"]["phase_grid_max_error"] < 1e-12
    assert results["power_normalization"]["relative_error"] < 1e-9
    assert results["eq14_statistics"]["relative_error"] < 0.01
    assert abs(results["pattern_peak"]["peak_db"] - 7.1) < 0.2
    assert results["q_star_full_refresh"]["passed"]
    assert results["gross_se_vs_snr"]["passed"]
    assert results["no_em_reproducibility"]["bitwise_equal"]


if __name__ == "__main__":
    main()
