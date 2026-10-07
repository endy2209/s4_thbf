"""
Baseline B4: Threshold-Based Dynamic THBF (Heuristic Event-Triggered THBF)
------------------------------------------------------------------
- Hardware: Tri-Hybrid Beamforming Array (Nt = 64, NRF = 4, NEM = 8, Ns = 4)
- Timescale Structure:
    * Dynamic Event-Triggered Updates based on Beam Gain Degradation Thresholds:
        - Trigger EM Update (u_EM = 1) if Beam Gain Drop > threshold_EM
        - Trigger RF Update (u_RF = 1) if Beam Gain Drop > threshold_RF
        - Otherwise, Freeze Hardware Configuration (u_EM = 0, u_RF = 0)
    * BB Layer: Instantaneous update at every slot t (0.125 ms)
- Costs: Pays realistic Pilot Overhead (c_pilot) and Switching Cost (c_sw)
- Purpose: Benchmarks heuristic rule-based dynamic scheduling against DRL.
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
from full_refresh import average_beam_gains, greedy_rf_full_refresh, select_q_star
from scheduling import greedy_user_selection
from s3 import S3Accounting
from dynamic_channel import load_or_generate_trajectory, DynamicTHBFConfig


def eval_current_config_gain(
    cfg: StaticTHBFConfig,
    aods: np.ndarray,
    q: int,
    beams: List[int]
) -> float:
    if not beams:
        return 0.0
    Gamma_q = average_beam_gains(cfg, aods, q)
    score = 0.0
    for b in beams:
        score += float(np.sum(Gamma_q[:, b - 1]))
    return score


def run_b4_simulation(
    trajectory_channels: np.ndarray,
    aod_trajectories: np.ndarray,
    thresh_EM: float = 0.20,
    thresh_RF: float = 0.10,
    cfg: StaticTHBFConfig = None
) -> Dict[str, Any]:
    if cfg is None:
        cfg = StaticTHBFConfig()

    if trajectory_channels is None or len(trajectory_channels) == 0:
        raise ValueError("trajectory_channels không thể để trống hoặc None.")

    if aod_trajectories is None:
        raise ValueError("Thiếu mảng góc AoD từ Stage 2 dataset! Baseline B4 yêu cầu aod_trajectories thực tế.")

    T_ep = len(trajectory_channels)

    em_codebook, _ = build_em_codebook(cfg)
    rf_codebook = build_rf_codebook(cfg.Nt, cfg.B, cfg.beta)

    b4_s3_cfg = S3Config(
        tau_bb=1.0,
        tau_rf=32.0,
        tau_em=64.0,
        lambda_rf=20.0,
        lambda_em=100.0,
        n_sym=14,
        t_rf=cfg.NRF * 10,
        t_em=cfg.NRF * 40,
        n_rf=cfg.NRF
    )
    accounting = S3Accounting(b4_s3_cfg)

    gross_se_hist = np.zeros(T_ep)
    net_se_hist = np.zeros(T_ep)
    utility_hist = np.zeros(T_ep)
    pilot_overhead_hist = np.zeros(T_ep)
    switching_cost_hist = np.zeros(T_ep)
    selected_q_hist = np.zeros(T_ep, dtype=int)

    num_EM_triggers = 0
    num_RF_triggers = 0

    current_q = 1
    current_beams = [1, 2, 3, 4]
    last_updated_gain = 0.0

    for t in range(T_ep):
        alpha_t = trajectory_channels[t]
        aod_t = aod_trajectories[t]

        if t == 0:
            u_EM = 1
            u_RF = 1
            q_star, b_star, scores, _ = select_q_star(cfg, aod_t)
            current_q = q_star
            current_beams = b_star.tolist()
            last_updated_gain = scores[q_star - 1]
            num_EM_triggers += 1
            num_RF_triggers += 1
            refresh_rf = True
        else:
            current_gain = eval_current_config_gain(cfg, aod_t, current_q, current_beams)

            if last_updated_gain > 1e-12:
                degradation = (last_updated_gain - current_gain) / last_updated_gain
            else:
                degradation = 0.0

            if degradation > thresh_EM:
                u_EM = 1
                u_RF = 1
                q_star, b_star, scores, _ = select_q_star(cfg, aod_t)
                current_q = q_star
                current_beams = b_star.tolist()
                last_updated_gain = scores[q_star - 1]
                num_EM_triggers += 1
                num_RF_triggers += 1
                refresh_rf = True
            elif degradation > thresh_RF:
                u_EM = 0
                u_RF = 1
                b_star, score_rf, _ = greedy_rf_full_refresh(cfg, aod_t, current_q)
                current_beams = b_star.tolist()
                last_updated_gain = score_rf
                num_RF_triggers += 1
                refresh_rf = True
            else:
                u_EM = 0
                u_RF = 0
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
        "EM_triggers_count": num_EM_triggers,
        "RF_triggers_count": num_RF_triggers,
        "selected_q_history": selected_q_hist,
    }

def grid_search_thresholds(
    val_channels: np.ndarray,
    val_aods: np.ndarray,
    grid_thresh_EM: List[float] = [0.02, 0.05, 0.1, 0.2],
    grid_thresh_RF: List[float] = [0.02, 0.05, 0.1, 0.2, 0.3],
    cfg: StaticTHBFConfig = None
) -> tuple[float, float, float]:
    best_utility = -1e9
    best_thresh_EM = grid_thresh_EM[0]
    best_thresh_RF = grid_thresh_RF[0]

    for thresh_em in grid_thresh_EM:
        for thresh_rf in grid_thresh_RF:
            res = run_b4_simulation(
                val_channels, val_aods,
                thresh_EM=thresh_em,
                thresh_RF=thresh_rf,
                cfg=cfg
            )
            
            avg_util = res["M3_Average_Utility"]
            if avg_util > best_utility:
                best_utility = avg_util
                best_thresh_EM = thresh_em
                best_thresh_RF = thresh_rf

    return best_thresh_EM, best_thresh_RF, best_utility
if __name__ == "__main__":
    print("=== Đang kiểm thử độc lập Baseline B4 (Threshold-Based Dynamic THBF) ===")
    dyn_cfg = DynamicTHBFConfig()
    traj, cache_path, reused = load_or_generate_trajectory(
        dyn_cfg, seed=20000, speeds_kmh=30.0, slots=200
    )
    trajectory_channels = traj["alpha"]
    aod_trajectories = traj["aods"]

    res = run_b4_simulation(trajectory_channels, aod_trajectories, thresh_EM=0.20, thresh_RF=0.10)
    print("\n--- KẾT QUẢ CHẠY B4 ---")
    for k, v in res.items():
        if k != "selected_q_history":
            print(f"  {k}: {v}")
    print("✅ Baseline B4 đã vận hành hoàn chỉnh!")
