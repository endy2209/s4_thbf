"""
Baseline B2: Fixed Tri-Timescale THBF (Periodic 160/40 Slots)
------------------------------------------------------------------
- Hardware: Tri-Hybrid Beamforming Array (Nt = 64, NRF = 4, NEM = 8, Ns = 4)
- Timescale Structure:
    * EM Layer: Periodic update every T_EM = 160 slots (20 ms)
    * RF Layer: Periodic update every T_RF = 40 slots (5 ms)
    * BB Layer: Instantaneous update at every slot t (0.125 ms)
- Costs: Pays realistic Pilot Overhead (c_pilot) and Hardware Switching Cost (c_sw)
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

sys.path.insert(0, "/workspace/scratch")

from config import StaticTHBFConfig, S3Config
from em import build_em_codebook
from rf import build_rf_codebook, rf_precoder
from channel import em_configured_channel_direct, effective_channels
from full_refresh import select_q_star, greedy_rf_full_refresh
from scheduling import greedy_user_selection
from s3 import S3Accounting
from dynamic_channel import load_or_generate_trajectory, DynamicTHBFConfig


def run_b2_simulation(
    trajectory_channels: np.ndarray,
    aod_trajectories: np.ndarray,
    T_EM: int = 160,
    T_RF: int = 40,
    cfg: StaticTHBFConfig = None
) -> Dict[str, Any]:
    if cfg is None:
        cfg = StaticTHBFConfig()

    if trajectory_channels is None or len(trajectory_channels) == 0:
        raise ValueError("trajectory_channels không thể để trống hoặc None.")

    if aod_trajectories is None:
        raise ValueError("Thiếu mảng góc AoD từ Stage 2 dataset! Baseline B2 yêu cầu aod_trajectories thực tế.")

    T_ep = len(trajectory_channels)

    em_codebook, _ = build_em_codebook(cfg)
    rf_codebook = build_rf_codebook(cfg.Nt, cfg.B, cfg.beta)

    b2_s3_cfg = S3Config(
        tau_bb=1.0,
        tau_rf=32.0,
        tau_em=64.0,
        lambda_rf=20.0,
        lambda_em=100.0,
        n_sym=14,
        t_rf=T_RF,
        t_em=T_EM,
        n_rf=cfg.NRF
    )
    accounting = S3Accounting(b2_s3_cfg)

    gross_se_hist = np.zeros(T_ep)
    net_se_hist = np.zeros(T_ep)
    utility_hist = np.zeros(T_ep)
    pilot_overhead_hist = np.zeros(T_ep)
    switching_cost_hist = np.zeros(T_ep)
    selected_q_hist = np.zeros(T_ep, dtype=int)

    current_q = 1
    current_beams = [1, 2, 3, 4]

    for t in range(T_ep):
        alpha_t = trajectory_channels[t]
        aod_t = aod_trajectories[t]

        u_EM = 1 if (t % T_EM == 0) else 0
        u_RF = 1 if (t % T_RF == 0) else 0

        if u_EM == 1:
            q_star, b_star, _, _ = select_q_star(cfg, aod_t)
            current_q = q_star
            current_beams = b_star.tolist()
            refresh_rf = True
        elif u_RF == 1:
            b_star, _, _ = greedy_rf_full_refresh(cfg, aod_t, current_q)
            current_beams = b_star.tolist()
            refresh_rf = True
        else:
            refresh_rf = False

        gq = em_codebook[current_q - 1]
        H_EM_q = em_configured_channel_direct(alpha_t, aod_t, gq, cfg.Nt)

        F_RF = rf_precoder(rf_codebook, current_beams)
        H_eff = effective_channels(H_EM_q, F_RF)

        selected_users, gross_se, gammas, F_BB, eta, alpha_rzf = greedy_user_selection(
            H_eff, F_RF, cfg.Ns, cfg.Pmax, cfg.sigma2
        )

        res = accounting.step(
            t=t,
            gross_se=gross_se,
            q=current_q,
            beams=current_beams,
            refresh_rf=refresh_rf
        )

        gross_se_hist[t] = res.gross_se
        net_se_hist[t] = res.net_se
        utility_hist[t] = res.utility
        pilot_overhead_hist[t] = res.pilot_total
        switching_cost_hist[t] = 100.0 * res.switch_em + 20.0 * res.switch_rf
        selected_q_hist[t] = current_q

    return {
        "M1_Gross_SE": float(np.mean(gross_se_hist)),
        "M2_Net_SE": float(np.mean(net_se_hist)),
        "M3_Average_Utility": float(np.mean(utility_hist)),
        "M4_Pilot_Overhead_Ratio": float(np.mean(pilot_overhead_hist)),
        "M6_Switching_Cost": float(np.sum(switching_cost_hist)),
        "selected_q_history": selected_q_hist,
    }


if __name__ == "__main__":
    print("=== Đang kiểm thử độc lập Baseline B2 (Fixed Tri-Timescale THBF 160/40) ===")
    dyn_cfg = DynamicTHBFConfig()
    traj, cache_path, reused = load_or_generate_trajectory(
        dyn_cfg, seed=20000, speeds_kmh=30.0, slots=200
    )
    trajectory_channels = traj["alpha"]
    aod_trajectories = traj["aods"]

    res = run_b2_simulation(trajectory_channels, aod_trajectories, T_EM=160, T_RF=40)
    print("\n--- KẾT QUẢ CHẠY B2 ---")
    for k, v in res.items():
        if k != "selected_q_history":
            print(f"  {k}: {v:.4f}")
    print("✅ Baseline B2 đã vận hành hoàn chỉnh!")
