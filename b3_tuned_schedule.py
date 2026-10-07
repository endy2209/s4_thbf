"""
Baseline B3: Tuned Periodic THBF (Grid-Search Optimized Periodic THBF)
------------------------------------------------------------------
- Hardware: Tri-Hybrid Beamforming Array (Nt = 64, NRF = 4, NEM = 8, Ns = 4)
- Timescale Structure:
    * EM Layer: Periodic update every T_EM* slots (Optimized offline via Grid Search)
    * RF Layer: Periodic update every T_RF* slots (Optimized offline via Grid Search)
    * BB Layer: Instantaneous update at every slot t (0.125 ms)
- Costs: Pays realistic Pilot Overhead (c_pilot) and Hardware Switching Cost (c_sw)
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
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


def run_b3_simulation(
    trajectory_channels: np.ndarray,
    aod_trajectories: np.ndarray,
    T_EM_opt: int = 160,
    T_RF_opt: int = 40,
    cfg: StaticTHBFConfig = None
) -> Dict[str, Any]:
    if cfg is None:
        cfg = StaticTHBFConfig()

    if trajectory_channels is None or len(trajectory_channels) == 0:
        raise ValueError("trajectory_channels không thể để trống hoặc None.")

    if aod_trajectories is None:
        raise ValueError("Thiếu mảng góc AoD từ Stage 2 dataset! Baseline B3 yêu cầu aod_trajectories thực tế.")

    T_ep = len(trajectory_channels)

    em_codebook, _ = build_em_codebook(cfg)
    rf_codebook = build_rf_codebook(cfg.Nt, cfg.B, cfg.beta)

    b3_s3_cfg = S3Config(
        tau_bb=1.0,
        tau_rf=32.0,
        tau_em=64.0,
        lambda_rf=20.0,
        lambda_em=100.0,
        n_sym=14,
        t_rf=T_RF_opt,
        t_em=T_EM_opt,
        n_rf=cfg.NRF
    )
    accounting = S3Accounting(b3_s3_cfg)

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

        u_EM = 1 if (t % T_EM_opt == 0) else 0
        u_RF = 1 if (t % T_RF_opt == 0) else 0

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
        "tuned_T_EM": T_EM_opt,
        "tuned_T_RF": T_RF_opt
    }


def grid_search_optimal_periods(
    val_channels: np.ndarray,
    val_aods: np.ndarray,
    candidate_T_EM: List[int] = [160, 320, 800, 1600, 4000],  # Cập nhật theo chuẩn {1, 2, 5, 10, 25} * 160
    candidate_T_RF: List[int] = [40, 80, 200, 400, 800, 2000],   # Cập nhật theo chuẩn {1, 2, 5, 10, 20, 50} * 40
    cfg: StaticTHBFConfig = None
) -> Tuple[int, int, float]:
    best_utility = -1e9
    best_T_EM = candidate_T_EM[0]
    best_T_RF = candidate_T_RF[0]

    for t_em in candidate_T_EM:
        for t_rf in candidate_T_RF:
            if t_em < t_rf:
                continue
            
            res = run_b3_simulation(
                val_channels, val_aods,
                T_EM_opt=t_em, T_RF_opt=t_rf,
                cfg=cfg
            )
            
            avg_util = res["M3_Average_Utility"]
            if avg_util > best_utility:
                best_utility = avg_util
                best_T_EM = t_em
                best_T_RF = t_rf

    return best_T_EM, best_T_RF, best_utility

if __name__ == "__main__":
    print("=== Đang kiểm thử độc lập Baseline B3 (Tuned Periodic THBF) ===")
    dyn_cfg = DynamicTHBFConfig()
    traj, cache_path, reused = load_or_generate_trajectory(
        dyn_cfg, seed=20000, speeds_kmh=30.0, slots=200
    )
    trajectory_channels = traj["alpha"]
    aod_trajectories = traj["aods"]

    best_EM, best_RF, best_util = grid_search_optimal_periods(
        trajectory_channels, aod_trajectories,
        candidate_T_EM=[80, 160],
        candidate_T_RF=[20, 40, 80]
    )
    print(f"👉 Kết quả Grid Search: T_EM* = {best_EM} slots, T_RF* = {best_RF} slots (Validation Utility = {best_util:.4f})")

    res = run_b3_simulation(trajectory_channels, aod_trajectories, T_EM_opt=best_EM, T_RF_opt=best_RF)
    print("\n--- KẾT QUẢ CHẠY B3 ---")
    for k, v in res.items():
        if k != "selected_q_history":
            print(f"  {k}: {v}")
    print("✅ Baseline B3 đã vận hành hoàn chỉnh!")
