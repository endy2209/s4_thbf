"""S2 acceptance checks and diagnostics.

The `effective_snr_*` diagnostic is explicitly PROVISIONAL while awaiting the
supervisor's definition. The temporary convention is

    gamma_eff = Pmax/(Ns*sigma2) * ||h_tilde||_2^2,

where h_tilde is the instantaneous effective channel after fixed EM/RF layers.
This is isolated in one function so it can be replaced without changing the
mobility/fading model.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.special import j0

from channel import em_configured_channel_direct, effective_channels
from dynamic_channel import (
    C_LIGHT,
    DynamicTHBFConfig,
    doppler_hz,
    empirical_lag_correlation,
    fading_correlation,
    generate_trajectory,
    load_or_generate_trajectory,
    propagate_with_reflection,
    speed_mps,
)
from em import build_em_codebook, wrap_angle
from full_refresh import average_beam_gains, select_q_star
from rf import build_rf_codebook, rf_precoder


def check_rho(cfg, slots=None, seed=2201):
    T = cfg.trajectory_slots if slots is None else int(slots)
    target_rounded = {3: 0.9991, 10: 0.9896, 30: 0.9087, 60: 0.6598}
    rows = {}
    for i, v in enumerate((3, 10, 30, 60)):
        traj = generate_trajectory(cfg, seed + i, speeds_kmh=float(v), slots=T)
        emp = empirical_lag_correlation(traj["alpha"], 1)[1]
        theory = float(fading_correlation(v, cfg.fc_hz, cfg.Ts))
        rows[v] = {
            "theory": theory,
            "target_4dp": target_rounded[v],
            "empirical": float(emp),
            "abs_error": abs(float(emp) - theory),
            "within_0p01": bool(abs(float(emp) - theory) <= 0.01),
        }
    theory_values = [rows[v]["theory"] for v in (3, 10, 30, 60)]
    return {
        "by_speed_kmh": rows,
        "ordering_ok": bool(np.all(np.diff(theory_values) < 0.0)),
        "all_within_0p01": all(rows[v]["within_0p01"] for v in rows),
        "passed": bool(np.all(np.diff(theory_values) < 0.0) and all(rows[v]["within_0p01"] for v in rows)),
    }


def check_fdTs(cfg):
    rows = {}
    for v in (3, 10, 30, 60):
        main = float(doppler_hz(v, cfg.fc_hz) * cfg.Ts)
        at_1ms = float(doppler_hz(v, cfg.fc_hz) * 1e-3)
        rows[v] = {"fdTs_0p125ms": main, "fdTs_1ms": at_1ms}
    return {
        "by_speed_kmh": rows,
        "all_default_le_0p383": all(rows[v]["fdTs_0p125ms"] <= 0.383 for v in rows),
        "30_and_60_violate_at_1ms": bool(rows[30]["fdTs_1ms"] > 0.383 and rows[60]["fdTs_1ms"] > 0.383),
        "passed": bool(all(rows[v]["fdTs_0p125ms"] <= 0.383 for v in rows) and rows[30]["fdTs_1ms"] > 0.383 and rows[60]["fdTs_1ms"] > 0.383),
    }


def check_reflection_geometry(cfg):
    # A deterministic stress case near the outer/radial boundaries, using many
    # slots so several reflections occur.
    traj = generate_trajectory(cfg, seed=2277, speeds_kmh=60.0, slots=cfg.trajectory_slots)
    p = traj["positions"]
    theta_c = float(traj["theta_c"])
    r = np.linalg.norm(p, axis=2)
    theta = np.arctan2(p[:, :, 1], p[:, :, 0])
    angular_dev = np.abs(wrap_angle(theta - theta_c))
    inside = bool(
        np.all(r >= cfg.r_min_m - 2e-8)
        and np.all(r <= cfg.r_max_m + 2e-8)
        and np.all(angular_dev <= cfg.sector_half_width + 2e-8)
    )

    # Heading must change iff a reflection occurs, up to numerical tolerance.
    dh = np.abs(wrap_angle(np.diff(traj["headings"], axis=0)))
    changed = dh > 1e-10
    reflected = traj["reflection_count"][1:] > 0
    # Tangential/measure-zero contacts could leave heading unchanged, but none
    # occur in this deterministic test; enforce the intended rule here.
    no_spurious = bool(np.all(~changed | reflected))
    all_reflections_change = bool(np.all(~reflected | changed))

    # Direct unit tests of speed preservation for each boundary type.
    speed0 = 10.0
    tests = []
    tc = 0.0
    # outer circle
    cases = [
        (np.array([99.999, 0.0]), np.array([speed0, 0.0])),
        (np.array([30.001, 0.0]), np.array([-speed0, 0.0])),
        (50.0 * np.array([np.cos(cfg.sector_half_width - 1e-5), np.sin(cfg.sector_half_width - 1e-5)]), np.array([0.0, speed0])),
        (50.0 * np.array([np.cos(-cfg.sector_half_width + 1e-5), np.sin(-cfg.sector_half_width + 1e-5)]), np.array([0.0, -speed0])),
    ]
    for p0, v0 in cases:
        _, v1, hits = propagate_with_reflection(p0, v0, 0.02, tc, cfg)
        tests.append({"hits": hits, "speed_error": abs(np.linalg.norm(v1) - np.linalg.norm(v0))})
    speed_preserved = all(x["speed_error"] < 1e-12 for x in tests)

    return {
        "always_inside_sector": inside,
        "total_reflections": int(np.sum(traj["reflection_count"])),
        "heading_changes": int(np.sum(changed)),
        "no_heading_change_without_reflection": no_spurious,
        "every_reflection_changes_heading_in_test": all_reflections_change,
        "speed_preserved_unit_tests": speed_preserved,
        "boundary_unit_tests": tests,
        "passed": bool(inside and no_spurious and all_reflections_change and speed_preserved),
    }


def check_offsets_fixed(cfg):
    traj = generate_trajectory(cfg, seed=2288, speeds_kmh=30.0, slots=2000)
    inferred = wrap_angle(traj["aods"] - traj["mean_aods"][:, :, None])
    ref = inferred[0]
    err = float(np.max(np.abs(wrap_angle(inferred - ref[None, :, :]))))
    return {"max_offset_drift_rad": err, "passed": bool(err < 1e-12)}


def check_trajectory_cache(cfg, cache_dir="data/trajectories"):
    seed = 2299
    traj1, path, reused1 = load_or_generate_trajectory(cfg, seed, speeds_kmh=30.0, cache_dir=cache_dir)
    traj2, path2, reused2 = load_or_generate_trajectory(cfg, seed, speeds_kmh=30.0, cache_dir=cache_dir)
    keys = ["positions", "headings", "reflection_count", "aod_offsets", "aods", "alpha", "rho"]
    equal = all(np.array_equal(np.asarray(traj1[k]), np.asarray(traj2[k])) for k in keys)
    return {
        "cache_path": str(path),
        "same_path": bool(path == path2),
        "first_call_reused_existing": bool(reused1),
        "second_call_reused_existing": bool(reused2),
        "bitwise_equal": bool(equal),
        "passed": bool(reused2 and equal),
    }


def beam_coherence_sanity(cfg):
    beamwidth = 0.886 * 2.0 / cfg.Nt
    times = {}
    for v in (3, 10, 30, 60):
        t = 50.0 * beamwidth / float(speed_mps(v))
        times[v] = float(t)
    targets = {3: 1.66, 10: 0.50, 30: 0.17, 60: 0.083}
    return {
        "half_power_beamwidth_rad": float(beamwidth),
        "half_power_beamwidth_deg": float(np.rad2deg(beamwidth)),
        "coherence_time_s": times,
        "targets_s": targets,
        "passed": all(abs(times[v] - targets[v]) <= (0.01 if v != 60 else 0.002) for v in times),
    }


def provisional_effective_snr_linear(cfg, gains_kl, aods_kl, q, beams):
    """PROVISIONAL diagnostic definition pending supervisor confirmation.

    gamma_eff,k = Pmax/(Ns*sigma2) * ||h_tilde_k||^2.
    This is NOT Eq. (22) post-RZF SINR.
    """
    gq = build_em_codebook(cfg)[0][q - 1]
    rf = build_rf_codebook(cfg.Nt, cfg.B, cfg.beta)
    F_RF = rf_precoder(rf, beams)
    H_em = em_configured_channel_direct(gains_kl, aods_kl, gq, cfg.Nt)
    H_eff = effective_channels(H_em, F_RF)
    return (cfg.Pmax / (cfg.Ns * cfg.sigma2)) * np.sum(np.abs(H_eff) ** 2, axis=1)


def _nearest_pattern_indices(theta, M):
    """Vectorized nearest grid index for Eq. (13)/(14)."""
    th = wrap_angle(np.asarray(theta, dtype=float))
    step = 2.0 * np.pi / M
    return (np.floor((th + np.pi) / step + 0.5).astype(int)) % M


def _user_fixed_beam_gamma_series(cfg, aods_tl, q, beam):
    """Vectorized Eq. (34) for one user and one fixed beam over time."""
    gq = build_em_codebook(cfg)[0][q - 1]
    fb = build_rf_codebook(cfg.Nt, cfg.B, cfg.beta)[beam - 1]
    aods_tl = np.asarray(aods_tl, dtype=float)
    T, L = aods_tl.shape
    out = np.zeros(T, dtype=float)
    n = np.arange(cfg.Nt)
    for ell in range(L):
        th = aods_tl[:, ell]
        g2 = gq[_nearest_pattern_indices(th, cfg.M)] ** 2
        # a(theta)^H f_b for all t.
        Aconj = np.exp(-1j * np.pi * np.sin(th)[:, None] * n[None, :]) / np.sqrt(cfg.Nt)
        proj2 = np.abs(Aconj @ fb) ** 2
        out += (cfg.Nt / cfg.L) * g2 * proj2
    return out


def _user_effective_snr_series(cfg, alpha_tl, aods_tl, q, beams):
    """Vectorized provisional effective-SNR series for one user."""
    gq = build_em_codebook(cfg)[0][q - 1]
    rf = build_rf_codebook(cfg.Nt, cfg.B, cfg.beta)
    F_RF = rf_precoder(rf, beams)
    alpha_tl = np.asarray(alpha_tl, dtype=np.complex128)
    aods_tl = np.asarray(aods_tl, dtype=float)
    T, L = aods_tl.shape
    n = np.arange(cfg.Nt)
    H_em = np.zeros((T, cfg.Nt), dtype=np.complex128)
    for ell in range(L):
        th = aods_tl[:, ell]
        gv = gq[_nearest_pattern_indices(th, cfg.M)]
        Aconj = np.exp(-1j * np.pi * np.sin(th)[:, None] * n[None, :]) / np.sqrt(cfg.Nt)
        H_em += np.conj(alpha_tl[:, ell])[:, None] * gv[:, None] * Aconj
    H_em *= np.sqrt(cfg.Nt / cfg.L)
    H_eff = H_em @ F_RF
    return (cfg.Pmax / (cfg.Ns * cfg.sigma2)) * np.sum(np.abs(H_eff) ** 2, axis=1)


def make_diagnostics(cfg, output_dir="figures/diagnostics", seed=2301, max_lag=200):
    """Create the S2 diagnostic plots requested by the manuscript."""
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    speeds = (3, 10, 30, 60)

    # Same seed => same center, initial positions/headings/offsets; only speed differs.
    trajectories = {v: generate_trajectory(cfg, seed, speeds_kmh=float(v)) for v in speeds}

    # Correlation diagnostics, one plot per speed for readability.
    corr_files = []
    lags = np.arange(max_lag + 1)
    for v in speeds:
        tr = trajectories[v]
        emp = empirical_lag_correlation(tr["alpha"][:, 0, 0], max_lag)
        rho = float(tr["rho"][0])
        fD = float(doppler_hz(v, cfg.fc_hz))
        fig, ax = plt.subplots(figsize=(7.0, 4.5))
        ax.plot(lags, emp, marker="o", markevery=max(1, max_lag // 20), label="empirical")
        ax.plot(lags, rho ** lags, label=r"$\rho^{lag}$")
        ax.plot(lags, j0(2.0 * np.pi * fD * cfg.Ts * lags), label=r"$J_0(2\pi f_D lag T_s)$")
        ax.set_xlabel("Lag [slots]")
        ax.set_ylabel("Correlation")
        ax.set_title(f"S2 fading correlation, {v} km/h")
        ax.grid(True, alpha=0.3)
        ax.legend()
        p = out / f"correlation_{v}kmh.png"
        fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
        corr_files.append(str(p))

    # Fix q and RF config from common t=0 AoDs. Same seed means identical t=0 AoDs.
    aods0 = trajectories[30]["aods"][0]
    q_star, beams, _, _ = select_q_star(cfg, aods0)
    fixed_beam = int(beams[0])

    # Average Gamma of one fixed user/beam versus time for each speed.
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    gamma_summary = {}
    for v in speeds:
        tr = trajectories[v]
        vals = _user_fixed_beam_gamma_series(cfg, tr["aods"][:, 0, :], q_star, fixed_beam)
        time_s = np.arange(cfg.trajectory_slots) * cfg.Ts
        ax.plot(time_s, vals, label=f"{v} km/h")
        gamma_summary[v] = {"min": float(vals.min()), "max": float(vals.max()), "mean": float(vals.mean())}
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$\Gamma_{k,t}(q,b)$")
    ax.set_title(f"Average beam gain vs time (user 1, q={q_star}, b={fixed_beam})")
    ax.grid(True, alpha=0.3); ax.legend()
    gamma_path = out / "average_beam_gain_vs_time.png"
    fig.tight_layout(); fig.savefig(gamma_path, dpi=150); plt.close(fig)

    # Provisional instantaneous effective SNR using a fixed EM/RF configuration.
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    eff_summary = {}
    for v in speeds:
        tr = trajectories[v]
        vals = _user_effective_snr_series(
            cfg, tr["alpha"][:, 0, :], tr["aods"][:, 0, :], q_star, beams
        )
        db = 10.0 * np.log10(np.maximum(vals, 1e-15))
        time_s = np.arange(cfg.trajectory_slots) * cfg.Ts
        ax.plot(time_s, db, label=f"{v} km/h")
        eff_summary[v] = {"min_db": float(db.min()), "max_db": float(db.max()), "mean_db": float(db.mean())}
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Provisional effective SNR [dB]")
    ax.set_title(f"PROVISIONAL effective SNR vs time (user 1, fixed q={q_star}, beams={beams.tolist()})")
    ax.grid(True, alpha=0.3); ax.legend()
    eff_path = out / "provisional_effective_snr_vs_time.png"
    fig.tight_layout(); fig.savefig(eff_path, dpi=150); plt.close(fig)

    return {
        "fixed_q": int(q_star),
        "fixed_beams": beams.tolist(),
        "fixed_beam_for_gamma": fixed_beam,
        "correlation_plots": corr_files,
        "average_gain_plot": str(gamma_path),
        "effective_snr_plot": str(eff_path),
        "effective_snr_definition": "PROVISIONAL: Pmax/(Ns*sigma2) * ||h_tilde_k||_2^2",
        "gamma_summary": gamma_summary,
        "effective_snr_summary": eff_summary,
    }

def main():
    cfg = DynamicTHBFConfig()
    results = {
        "rho_validation": check_rho(cfg),
        "fdTs_validation": check_fdTs(cfg),
        "reflection_geometry": check_reflection_geometry(cfg),
        "fixed_aod_offsets": check_offsets_fixed(cfg),
        "trajectory_cache": check_trajectory_cache(cfg),
        "beam_coherence": beam_coherence_sanity(cfg),
    }
    diagnostics = make_diagnostics(cfg)

    for name, value in results.items():
        print(f"[{name}]")
        print(value)
        print()
    print("[diagnostics]")
    print(diagnostics)
    print()

    assert all(x["passed"] for x in results.values())


if __name__ == "__main__":
    main()
