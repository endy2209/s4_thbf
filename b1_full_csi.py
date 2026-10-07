"""
Baseline B1: Perfect CSI Single-timescale THBF (Ideal THBF Upper Bound)
------------------------------------------------------------------------
- Hardware: Tri-Hybrid Beamforming Array (Nt = 64, NRF = 4, NEM = 8, Ns = 4)
- CSI: Perfect instantaneous CSI at every single slot t
- Decision: Continuous update of EM state q_t* and RF configuration b_t* at EVERY slot t
- Costs: Zero switching cost and zero pilot overhead (Ideal THBF upper bound)
"""

import sys
from pathlib import Path
from typing import Dict, Any
import numpy as np

sys.path.insert(0, "/workspace/scratch")

from config import StaticTHBFConfig, S3Config
from em import build_em_codebook
from rf import build_rf_codebook, rf_precoder
from channel import em_configured_channel_direct, effective_channels
from full_refresh import select_q_star
from scheduling import greedy_user_selection
from precoding import rzf_precoder
from s3 import S3Accounting
from dynamic_channel import load_or_generate_trajectory, DynamicTHBFConfig


def run_b1_simulation(
    trajectory_channels: np.ndarray,
    aod_trajectories: np.ndarray,
    cfg: StaticTHBFConfig = None
) -> Dict[str, Any]:
    if cfg is None:
        cfg = StaticTHBFConfig()

    if trajectory_channels is None or len(trajectory_channels) == 0:
        raise ValueError("trajectory_channels không thể để trống hoặc None.")
        
    if aod_trajectories is None:
        raise ValueError("Thiếu mảng góc AoD từ Stage 2 dataset! Baseline B1 yêu cầu aod_trajectories thực tế.")

    T_ep = len(trajectory_channels)
    
    em_codebook, _ = build_em_codebook(cfg)
    rf_codebook = build_rf_codebook(cfg.Nt, cfg.B, cfg.beta)
    
    b1_s3_cfg = S3Config(
        tau_bb=0, tau_rf=0, tau_em=0,
        lambda_rf=0.0, lambda_em=0.0,
        n_sym=14, t_rf=1, t_em=1, n_rf=cfg.NRF
    )
    accounting = S3Accounting(b1_s3_cfg)

    gross_se_hist = np.zeros(T_ep)
    net_se_hist = np.zeros(T_ep)
    utility_hist = np.zeros(T_ep)
    selected_q_hist = np.zeros(T_ep, dtype=int)

    for t in range(T_ep):
        alpha_t = trajectory_channels[t]
        aod_t = aod_trajectories[t]

        q_star, b_star, _, _ = select_q_star(cfg, aod_t)
        
        gq = em_codebook[q_star - 1]
        H_EM_q = em_configured_channel_direct(alpha_t, aod_t, gq, cfg.Nt)
        
        F_RF = rf_precoder(rf_codebook, b_star)
        H_eff = effective_channels(H_EM_q, F_RF)
        
        selected_users, gross_se, gammas, F_BB, eta, alpha_rzf = greedy_user_selection(
            H_eff, F_RF, cfg.Ns, cfg.Pmax, cfg.sigma2
        )
        
        res = accounting.step(t, gross_se, q_star, b_star, refresh_rf=True)

        gross_se_hist[t] = res.gross_se
        net_se_hist[t] = res.net_se
        utility_hist[t] = res.utility
        selected_q_hist[t] = q_star

    return {
        "M1_Gross_SE": float(np.mean(gross_se_hist)),
        "M2_Net_SE": float(np.mean(net_se_hist)),
        "M3_Average_Utility": float(np.mean(utility_hist)),
        "M4_Pilot_Overhead_Ratio": 0.0,
        "M6_Switching_Cost": 0.0,
        "selected_q_history": selected_q_hist,
    }


if __name__ == "__main__":
    print("=== Đang kiểm thử độc lập Baseline B1 (Full-CSI Single-timescale THBF) ===")
    dyn_cfg = DynamicTHBFConfig()
    traj, cache_path, reused = load_or_generate_trajectory(
        dyn_cfg, seed=20000, speeds_kmh=30.0, slots=200
    )
    trajectory_channels = traj["alpha"]
    aod_trajectories = traj["aods"]

    res = run_b1_simulation(trajectory_channels, aod_trajectories)
    print("\n--- KẾT QUẢ CHẠY B1 ---")
    for k, v in res.items():
        if k != "selected_q_history":
            print(f"  {k}: {v:.4f}")
    print("✅ Baseline B1 đã vận hành hoàn chỉnh!")
