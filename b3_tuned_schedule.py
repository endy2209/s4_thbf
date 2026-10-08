import os
import json
import numpy as np
import scipy.io as sio

# ==============================================================================
# 1. CONFIGURATION & ACCOUNTING (S3 SYSTEM MODEL)
# ==============================================================================
class S3Config:
    def __init__(self, tau_bb=1.0, tau_rf=32.0, tau_em=64.0, 
                 lambda_rf=20.0, lambda_em=50.0, n_sym=14, 
                 t_rf=40, t_em=160, n_rf=4, n_tx=64, n_users=4):
        self.tau_bb = tau_bb
        self.tau_rf = tau_rf
        self.tau_em = tau_em
        self.lambda_rf = lambda_rf
        self.lambda_em = lambda_em
        self.n_sym = n_sym
        self.t_rf = t_rf
        self.t_em = t_em
        self.n_rf = n_rf
        self.n_tx = n_tx
        self.n_users = n_users

class S3Accounting:
    def __init__(self, cfg: S3Config):
        self.cfg = cfg
        self.reset()
        
    def reset(self):
        self.prev_q = None
        self.prev_b = None
        
    def step(self, t, u_rf, u_em, current_q, current_b):
        """
        Calculates pilot overhead ratio c_pilot,t and hardware switching cost c_sw,t.
        Note: t = 0 initialization cost is EXCLUDED per paper specification.
        """
        # Exclude t = 0 initialization cost
        if t == 0:
            self.prev_q = current_q
            self.prev_b = np.copy(current_b) if current_b is not None else None
            return 0.0, 0.0
        
        # Pilot Overhead Ratio c_pilot,t distributed evenly over T_RF block
        c_pilot = self.cfg.tau_bb / self.cfg.n_sym
        if u_rf == 1 or u_em == 1:
            c_pilot += (u_rf * self.cfg.tau_rf + u_em * self.cfg.tau_em) / (self.cfg.t_rf * self.cfg.n_sym)
            
        # Hardware Switching Cost c_sw,t
        c_em = 1.0 if (self.prev_q is not None and current_q != self.prev_q) else 0.0
        if self.prev_b is not None and current_b is not None:
            c_rf = np.mean(current_b != self.prev_b)
        else:
            c_rf = 0.0
            
        c_sw = self.cfg.lambda_em * c_em + self.cfg.lambda_rf * c_rf
        
        # Update state history
        self.prev_q = current_q
        self.prev_b = np.copy(current_b) if current_b is not None else None
        
        return c_pilot, c_sw

# ==============================================================================
# 2. DATASET LOADER & SYNTHETIC CHANNEL TRAJECTORY GENERATOR
# ==============================================================================
class DatasetLoader:
    """
    Handles loading real .mat / .npy trajectory files if present,
    or generates realistic dynamic 3GPP mmWave channels with user mobility.
    """
    def __init__(self, data_dir="./data"):
        self.data_dir = data_dir
        
    def load_trajectory(self, seed=42, n_slots=1000, n_tx=64, n_users=4):
        mat_path = os.path.join(self.data_dir, f"trajectory_seed_{seed}.mat")
        npy_path = os.path.join(self.data_dir, f"trajectory_seed_{seed}.npy")
        
        if os.path.exists(mat_path):
            data = sio.loadmat(mat_path)
            return data['H_seq'], data['aod_seq']
        elif os.path.exists(npy_path):
            data = np.load(npy_path, allow_pickle=True).item()
            return data['H_seq'], data['aod_seq']
        else:
            # Generate realistic dynamic channel with user mobility & channel aging
            np.random.seed(seed)
            aod_seq = np.zeros((n_slots, n_users))
            H_seq = np.zeros((n_slots, n_users, n_tx), dtype=complex)
            
            initial_aod = np.linspace(-35.0, 35.0, n_users) + np.random.uniform(-5, 5, n_users)
            velocities = np.random.uniform(0.02, 0.08, n_users) # deg / slot
            
            for t in range(n_slots):
                aods = initial_aod + velocities * t
                aod_seq[t, :] = aods
                for k in range(n_users):
                    theta_rad = np.deg2rad(aods[k])
                    steering = np.exp(-1j * np.pi * np.arange(n_tx) * np.sin(theta_rad))
                    gain = (np.random.randn() + 1j * np.random.randn()) / np.sqrt(2)
                    H_seq[t, k, :] = gain * steering
            return H_seq, aod_seq

# ==============================================================================
# 3. THBF CODEBOOK & RZF PRECODER
# ==============================================================================
class THBFSystem:
    def __init__(self, n_tx=64, n_rf=4, n_users=4, n_em_codebook=8):
        self.n_tx = n_tx
        self.n_rf = n_rf
        self.n_users = n_users
        self.n_em_codebook = n_em_codebook
        self.em_tilts = np.linspace(-40.0, 40.0, n_em_codebook)
        
    def get_em_pattern(self, q_idx):
        tilt_deg = self.em_tilts[q_idx]
        tilt_rad = np.deg2rad(tilt_deg)
        # Antenna element response g_q(theta)
        return np.exp(-1j * np.pi * np.arange(self.n_tx) * np.sin(tilt_rad))
        
    def select_rf_beams(self, H_eff_t):
        # Select best RF steering vectors for scheduled users
        b_indices = []
        for k in range(self.n_users):
            h_k = H_eff_t[k, :]
            # Quantize phase to RF codebook
            b_k = np.exp(1j * np.angle(h_k)) / np.sqrt(self.n_tx)
            b_indices.append(b_k)
        return np.array(b_indices) # Shape: (n_rf, n_tx)
        
    def compute_rzf_precoder(self, H_eff_t, F_rf, snr_linear=10.0):
        # Effective channel through RF/EM layers
        H_eq = H_eff_t @ F_rf.T # (n_users, n_rf)
        # RZF Formula: W_bb = H_eq^H (H_eq H_eq^H + (K / SNR) I)^-1
        K = H_eq.shape[0]
        alpha = K / snr_linear
        inv_part = np.linalg.inv(H_eq @ H_eq.conj().T + alpha * np.eye(K))
        W_bb = H_eq.conj().T @ inv_part
        # Normalize columns
        norm = np.linalg.norm(W_bb, axis=0, keepdims=True)
        norm[norm == 0] = 1.0
        W_bb = W_bb / norm
        return W_bb

    def compute_instantaneous_rate(self, H_t, q_idx, F_rf, W_bb, snr_linear=10.0):
        g_em = self.get_em_pattern(q_idx)
        H_eff = H_t * g_em[None, :] # Apply EM pattern
        H_eq = H_eff @ F_rf.T # (K, N_RF)
        H_total = H_eq @ W_bb # (K, K)
        
        rates = []
        for k in range(self.n_users):
            signal = np.abs(H_total[k, k])**2
            interference = np.sum(np.abs(H_total[k, :])**2) - signal
            sinr = signal / (interference + 1.0 / snr_linear)
            rate_k = np.log2(1.0 + sinr)
            rates.append(rate_k)
        return np.sum(rates)

# ==============================================================================
# 4. BASELINE B3: TUNED PERIODIC GRID SEARCH & EVALUATION
# ==============================================================================
def run_b3_simulation(H_seq, aod_seq, cfg: S3Config, t_rf, t_em, thbf: THBFSystem):
    n_slots = H_seq.shape[0]
    accounting = S3Accounting(cfg)
    accounting.cfg.t_rf = t_rf
    accounting.cfg.t_em = t_em
    
    current_q = 0
    current_F_rf = None
    current_W_bb = None
    current_b_idx = None
    
    gross_se_hist = []
    net_se_hist = []
    utility_hist = []
    c_pilot_hist = []
    c_sw_hist = []
    
    for t in range(n_slots):
        # Determine periodic update triggers
        u_em = 1 if (t % t_em == 0) else 0
        u_rf = 1 if (t % t_rf == 0 or u_em == 1) else 0 # u_em <= u_rf constraint enforced
        
        # Hardware re-configuration if triggered
        if u_em == 1 or current_q is None:
            # Select best EM tilt for current AoD
            mean_aod = np.mean(aod_seq[t, :])
            current_q = np.argmin(np.abs(thbf.em_tilts - mean_aod))
            
        g_em = thbf.get_em_pattern(current_q)
        H_eff_t = H_seq[t] * g_em[None, :]
        
        if u_rf == 1 or current_F_rf is None:
            current_F_rf = thbf.select_rf_beams(H_eff_t)
            current_b_idx = np.angle(current_F_rf)
            
        current_W_bb = thbf.compute_rzf_precoder(H_eff_t, current_F_rf)
        
        # Measure transmission performance
        gross_rate = thbf.compute_instantaneous_rate(H_seq[t], current_q, current_F_rf, current_W_bb)
        c_pilot, c_sw = accounting.step(t, u_rf, u_em, current_q, current_b_idx)
        
        net_rate = (1.0 - c_pilot) * gross_rate
        utility = net_rate - c_sw
        
        gross_se_hist.append(gross_rate)
        net_se_hist.append(net_rate)
        utility_hist.append(utility)
        c_pilot_hist.append(c_pilot)
        c_sw_hist.append(c_sw)
        
    metrics = {
        "M1_Gross_SE": float(np.mean(gross_se_hist)),
        "M2_Net_SE": float(np.mean(net_se_hist)),
        "M3_Utility": float(np.mean(utility_hist)),
        "M4_Pilot_Overhead_Ratio": float(np.mean(c_pilot_hist)),
        "M6_Switching_Cost": float(np.mean(c_sw_hist)) # Corrected np.mean
    }
    return metrics

def grid_search_b3(val_H_seq, val_aod_seq, cfg: S3Config, thbf: THBFSystem):
    t_rf_candidates = [5, 10, 20, 40, 80, 160]
    t_em_candidates = [20, 40, 80, 160, 320, 640]
    
    best_utility = -float('inf')
    best_t_rf = None
    best_t_em = None
    search_results = []
    
    print("=== STARTING B3 GRID SEARCH (OBJECTIVE: MAXIMIZE MEAN VALIDATION UTILITY \\bar{U}) ===")
    for t_rf in t_rf_candidates:
        for t_em in t_em_candidates:
            if t_em < t_rf: # Enforce multi-timescale hierarchy T_EM >= T_RF
                continue
            res = run_b3_simulation(val_H_seq, val_aod_seq, cfg, t_rf, t_em, thbf)
            val_utility = res["M3_Utility"]
            search_results.append((t_rf, t_em, val_utility, res["M2_Net_SE"]))
            print(f"Candidate (T_RF={t_rf:3d}, T_EM={t_em:3d}) -> Mean Val Utility \\bar{{U}} = {val_utility:8.4f} bit/s/Hz | Net SE = {res['M2_Net_SE']:8.4f}")
            
            if val_utility > best_utility:
                best_utility = val_utility
                best_t_rf = t_rf
                best_t_em = t_em
                
    print("\n================================================================================")
    print(f">>> OPTIMAL B3 PERIODS: T_RF* = {best_t_rf}, T_EM* = {best_t_em} with Peak \\bar{{U}}* = {best_utility:.4f} bit/s/Hz <<<")
    print("================================================================================\n")
    return best_t_rf, best_t_em

# ==============================================================================
# 5. MAIN EXECUTION PIPELINE
# ==============================================================================
if __name__ == "__main__":
    cfg = S3Config()
    loader = DatasetLoader()
    thbf = THBFSystem(n_tx=cfg.n_tx, n_rf=cfg.n_rf, n_users=cfg.n_users)
    
    # Load Validation & Test trajectories
    val_H, val_aod = loader.load_trajectory(seed=101, n_slots=1000)
    test_H, test_aod = loader.load_trajectory(seed=202, n_slots=2000)
    
    # 1. Grid Search on Validation Set
    opt_t_rf, opt_t_em = grid_search_b3(val_H, val_aod, cfg, thbf)
    
    # 2. Final Evaluation on Test Set
    test_metrics = run_b3_simulation(test_H, test_aod, cfg, opt_t_rf, opt_t_em, thbf)
    print("=== FINAL TEST EVALUATION METRICS (BASELINE B3) ===")
    for k, v in test_metrics.items():
        print(f"  {k:25s}: {v:.6f}")
        
    # Output results to JSON
    out_dir = "/workspace/scratch"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "b3_results.json"), "w") as f:
        json.dump({"optimal_periods": {"t_rf": opt_t_rf, "t_em": opt_t_em}, "metrics": test_metrics}, f, indent=4)
