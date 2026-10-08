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
        if t == 0:
            self.prev_q = current_q
            self.prev_b = np.copy(current_b) if current_b is not None else None
            return 0.0, 0.0
        
        c_pilot = self.cfg.tau_bb / self.cfg.n_sym
        if u_rf == 1 or u_em == 1:
            c_pilot += (u_rf * self.cfg.tau_rf + u_em * self.cfg.tau_em) / (self.cfg.t_rf * self.cfg.n_sym)
            
        c_em = 1.0 if (self.prev_q is not None and current_q != self.prev_q) else 0.0
        if self.prev_b is not None and current_b is not None:
            c_rf = np.mean(current_b != self.prev_b)
        else:
            c_rf = 0.0
            
        c_sw = self.cfg.lambda_em * c_em + self.cfg.lambda_rf * c_rf
        
        self.prev_q = current_q
        self.prev_b = np.copy(current_b) if current_b is not None else None
        
        return c_pilot, c_sw

# ==============================================================================
# 2. DATASET LOADER & SYNTHETIC CHANNEL TRAJECTORY GENERATOR
# ==============================================================================
class DatasetLoader:
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
            np.random.seed(seed)
            aod_seq = np.zeros((n_slots, n_users))
            H_seq = np.zeros((n_slots, n_users, n_tx), dtype=complex)
            
            initial_aod = np.linspace(-35.0, 35.0, n_users) + np.random.uniform(-5, 5, n_users)
            velocities = np.random.uniform(0.02, 0.08, n_users)
            
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
        return np.exp(-1j * np.pi * np.arange(self.n_tx) * np.sin(tilt_rad))
        
    def select_rf_beams(self, H_eff_t):
        b_indices = []
        for k in range(self.n_users):
            h_k = H_eff_t[k, :]
            b_k = np.exp(1j * np.angle(h_k)) / np.sqrt(self.n_tx)
            b_indices.append(b_k)
        return np.array(b_indices)
        
    def compute_rzf_precoder(self, H_eff_t, F_rf, snr_linear=10.0):
        H_eq = H_eff_t @ F_rf.T
        K = H_eq.shape[0]
        alpha = K / snr_linear
        inv_part = np.linalg.inv(H_eq @ H_eq.conj().T + alpha * np.eye(K))
        W_bb = H_eq.conj().T @ inv_part
        norm = np.linalg.norm(W_bb, axis=0, keepdims=True)
        norm[norm == 0] = 1.0
        W_bb = W_bb / norm
        return W_bb

    def compute_instantaneous_rate(self, H_t, q_idx, F_rf, W_bb, snr_linear=10.0):
        g_em = self.get_em_pattern(q_idx)
        H_eff = H_t * g_em[None, :]
        H_eq = H_eff @ F_rf.T
        H_total = H_eq @ W_bb
        
        rates = []
        for k in range(self.n_users):
            signal = np.abs(H_total[k, k])**2
            interference = np.sum(np.abs(H_total[k, :])**2) - signal
            sinr = signal / (interference + 1.0 / snr_linear)
            rates.append(np.log2(1.0 + sinr))
        return np.sum(rates)

# ==============================================================================
# 4. BASELINE B4: THRESHOLD-TRIGGERED DYNAMIC SCHEDULING & EVALUATION
# ==============================================================================
def run_b4_simulation(H_seq, aod_seq, cfg: S3Config, zeta_rf, zeta_em, thbf: THBFSystem, epsilon=1e-12):
    n_slots = H_seq.shape[0]
    t_rf_block = cfg.t_rf # Candidate events occur every T_RF slots (default 40)
    n_blocks = n_slots // t_rf_block
    
    accounting = S3Accounting(cfg)
    
    current_q = 0
    current_F_rf = None
    current_W_bb = None
    current_b_idx = None
    
    e_ref = 0
    R_ref_gross = 0.0
    R_prev_gross = 0.0
    
    gross_se_hist = []
    net_se_hist = []
    utility_hist = []
    c_pilot_hist = []
    c_sw_hist = []
    
    for e in range(n_blocks):
        block_start = e * t_rf_block
        
        # Calculate real Gross SE degradation zeta_e at block start
        if e == 0:
            u_em = 1
            u_rf = 1
        else:
            zeta_e = max(0.0, (R_ref_gross - R_prev_gross) / max(R_ref_gross, epsilon))
            if zeta_e >= zeta_em:
                u_em = 1
                u_rf = 1
                e_ref = e
            elif zeta_e >= zeta_rf:
                u_em = 0
                u_rf = 1
                e_ref = e
            else:
                u_em = 0
                u_rf = 0
                
        block_gross_rates = []
        
        for t_offset in range(t_rf_block):
            t = block_start + t_offset
            
            # Update configuration if triggered at block start
            if t_offset == 0:
                if u_em == 1 or current_q is None:
                    mean_aod = np.mean(aod_seq[t, :])
                    current_q = np.argmin(np.abs(thbf.em_tilts - mean_aod))
                    
                g_em = thbf.get_em_pattern(current_q)
                H_eff_t = H_seq[t] * g_em[None, :]
                
                if u_rf == 1 or current_F_rf is None:
                    current_F_rf = thbf.select_rf_beams(H_eff_t)
                    current_b_idx = np.angle(current_F_rf)
                    
                current_W_bb = thbf.compute_rzf_precoder(H_eff_t, current_F_rf)
            else:
                # Inside block: only Baseband update
                g_em = thbf.get_em_pattern(current_q)
                H_eff_t = H_seq[t] * g_em[None, :]
                current_W_bb = thbf.compute_rzf_precoder(H_eff_t, current_F_rf)
                
            # Measure performance
            gross_rate = thbf.compute_instantaneous_rate(H_seq[t], current_q, current_F_rf, current_W_bb)
            c_pilot, c_sw = accounting.step(t, u_rf if t_offset == 0 else 0, u_em if t_offset == 0 else 0, current_q, current_b_idx)
            
            net_rate = (1.0 - c_pilot) * gross_rate
            utility = net_rate - c_sw
            
            gross_se_hist.append(gross_rate)
            net_se_hist.append(net_rate)
            utility_hist.append(utility)
            c_pilot_hist.append(c_pilot)
            c_sw_hist.append(c_sw)
            block_gross_rates.append(gross_rate)
            
        R_prev_gross = np.mean(block_gross_rates)
        if e == e_ref:
            R_ref_gross = R_prev_gross
            
    metrics = {
        "M1_Gross_SE": float(np.mean(gross_se_hist)),
        "M2_Net_SE": float(np.mean(net_se_hist)),
        "M3_Utility": float(np.mean(utility_hist)),
        "M4_Pilot_Overhead_Ratio": float(np.mean(c_pilot_hist)),
        "M6_Switching_Cost": float(np.mean(c_sw_hist)) # Corrected np.mean
    }
    return metrics

def grid_search_b4(val_H_seq, val_aod_seq, cfg: S3Config, thbf: THBFSystem):
    zeta_rf_candidates = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]
    zeta_em_candidates = [0.05, 0.10, 0.15, 0.25, 0.35]
    
    best_utility = -float('inf')
    best_zeta_rf = None
    best_zeta_em = None
    
    print("=== STARTING B4 THRESHOLD GRID SEARCH (OBJECTIVE: MAXIMIZE MEAN VALIDATION UTILITY \\bar{U}) ===")
    for z_rf in zeta_rf_candidates:
        for z_em in zeta_em_candidates:
            if z_em < z_rf: # EM threshold >= RF threshold
                continue
            res = run_b4_simulation(val_H_seq, val_aod_seq, cfg, z_rf, z_em, thbf)
            val_utility = res["M3_Utility"]
            print(f"Candidate (zeta_RF={z_rf:.2f}, zeta_EM={z_em:.2f}) -> Mean Val Utility \\bar{{U}} = {val_utility:8.4f} bit/s/Hz | Net SE = {res['M2_Net_SE']:8.4f}")
            
            if val_utility > best_utility:
                best_utility = val_utility
                best_zeta_rf = z_rf
                best_zeta_em = z_em
                
    print("\n================================================================================")
    print(f">>> OPTIMAL B4 THRESHOLDS: zeta_RF* = {best_zeta_rf:.2f}, zeta_EM* = {best_zeta_em:.2f} with Peak \\bar{{U}}* = {best_utility:.4f} bit/s/Hz <<<")
    print("================================================================================\n")
    return best_zeta_rf, best_zeta_em

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
    opt_z_rf, opt_z_em = grid_search_b4(val_H, val_aod, cfg, thbf)
    
    # 2. Final Evaluation on Test Set
    test_metrics = run_b4_simulation(test_H, test_aod, cfg, opt_z_rf, opt_z_em, thbf)
    print("=== FINAL TEST EVALUATION METRICS (BASELINE B4) ===")
    for k, v in test_metrics.items():
        print(f"  {k:25s}: {v:.6f}")
        
    # Output results to JSON
    out_dir = "/workspace/scratch"
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "b4_results.json"), "w") as f:
        json.dump({"optimal_thresholds": {"zeta_rf": opt_z_rf, "zeta_em": opt_z_em}, "metrics": test_metrics}, f, indent=4)
