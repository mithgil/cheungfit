# src/ivs_tools/physics/sliding_window_fitter_debug.py
"""
DEBUG VERSION: 加入即時 print 輸出的 SlidingWindowCheungFitter
用於診斷為什麼 fitting 失敗
"""
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List

# 複製 FittingResult 和 FittingConfig 從原檔案
@dataclass
class FittingConfig:
    window_size: int = 50
    step_size: int = 5
    min_r_squared: float = 0.98
    n_min: float = 1.0
    n_max: float = 2.5
    Rs_min: float = 0.0
    Rs_max: float = 1e10
    savgol_window: int = 21
    savgol_polyorder: int = 3
    current_threshold: float = 1e-12
    layer_type: str = "multilayer"
    
    def update_for_layer(self, layer_type: str):
        if layer_type == "monolayer":
            self.window_size = 30
            self.n_max = 2.0
            self.min_r_squared = 0.99
        elif layer_type == "bilayer":
            self.window_size = 40
            self.n_max = 2.2

@dataclass
class FittingResult:
    success: bool
    n: Optional[float] = None
    Rs: Optional[float] = None
    phi_b: Optional[float] = None
    r_squared: Optional[float] = None
    fit_range: Optional[Tuple[float, float]] = None
    fit_indices: Optional[Tuple[int, int]] = None
    score: float = 0.0
    warning: Optional[str] = None
    diagnostic: Dict = field(default_factory=dict)


class SlidingWindowCheungFitterDebug:
    """DEBUG 版本：每步都 print"""
    
    def __init__(self, config: Optional[FittingConfig] = None, verbose: bool = True):
        self.config = config or FittingConfig()
        self.q = 1.6021766e-19
        self.verbose = verbose  # ✅ 控制是否輸出 debug 訊息
        
    def _log(self, msg: str):
        if self.verbose:
            print(f"   [DEBUG] {msg}")
    
    def fit(self, bias: np.ndarray, current: np.ndarray, 
            area_cm2: float, T: float = 300, a_star: float = 54.0,
            layer_type: Optional[str] = None, verbose: Optional[bool] = None) -> FittingResult:
        
        if verbose is not None:
            self.verbose = verbose
            
        if layer_type:
            self.config.update_for_layer(layer_type)
            self._log(f"Layer override: window_size={self.config.window_size}, n_max={self.config.n_max}")
        
        # === 預處理 ===
        self._log(f"Starting preprocessing...")
        I_abs = np.abs(current)
        mask_valid = I_abs > self.config.current_threshold
        self._log(f"  Valid points: {np.sum(mask_valid)}/{len(I_abs)} (threshold={self.config.current_threshold:.2e} A)")
        
        if np.sum(mask_valid) < 10:
            self._log("❌ Preprocess fail: not enough valid points")
            return FittingResult(success=False, warning=f"有效數據點不足：{np.sum(mask_valid)} < 10")
        
        bias_proc = bias[mask_valid]
        I_proc = I_abs[mask_valid]
        
        # Savitzky-Golay
        window = min(self.config.savgol_window, len(I_proc) - 2)
        if window % 2 == 0: window += 1
        I_smooth = savgol_filter(I_proc, window_length=window, polyorder=self.config.savgol_polyorder)
        self._log(f"  After smoothing: {len(I_smooth)} points")
        
        # === 計算 Cheung Function I ===
        self._log("Computing Cheung Function I: dV/d(lnI)...")
        ln_I = np.log(I_smooth + 1e-20)
        dV_dlnI = np.gradient(bias_proc, ln_I)
        V_T = (1.380649e-23 * T) / self.q
        self._log(f"  V_T = {V_T:.4f} V")
        
        constant_term = area_cm2 * a_star * (T**2)
        self._log(f"  Constant term (A*A**T²) = {constant_term:.3e}")
        
        # === 滑動窗口掃描 ===
        self._log(f"Starting sliding window scan...")
        window_size = self.config.window_size
        step = self.config.step_size
        total_possible = (len(bias_proc) - window_size) // step + 1
        self._log(f"  Data length: {len(bias_proc)}, window: {window_size}, step: {step}")
        self._log(f"  Total possible windows: {total_possible}")
        
        best_result = FittingResult(success=False)
        stats_counter = {
            'total': 0, 'monotonic_fail': 0, 'r2_pre_fail': 0, 
            'constraint_fail': 0, 'r2_final_fail': 0, 'rs_mismatch': 0, 'success': 0
        }
        n_samples = []  # 收集部分 n 值用於診斷
        
        for start_idx in range(0, len(bias_proc) - window_size + 1, step):
            end_idx = start_idx + window_size
            stats_counter['total'] += 1
            
            I_win = I_smooth[start_idx:end_idx]
            V_win = bias_proc[start_idx:end_idx]
            y_win = dV_dlnI[start_idx:end_idx]
            
            # 檢查 1: 單調性
            if not np.all(np.diff(I_win) > 0):
                stats_counter['monotonic_fail'] += 1
                continue
            
            # 檢查 2: Cheung I 線性擬合
            if len(I_win) < 5:
                continue
            slope_a, intercept_a, r_value, _, _ = stats.linregress(I_win, y_win)
            r2_a = r_value**2
            n_ext = intercept_a / V_T
            Rs_ext = slope_a
            
            n_samples.append(n_ext)  # 收集用於診斷
            
            if r2_a < 0.9:
                stats_counter['r2_pre_fail'] += 1
                continue
            
            # 檢查 3: 物理約束
            if not (self.config.n_min < n_ext < self.config.n_max):
                stats_counter['constraint_fail'] += 1
                continue
            if not (self.config.Rs_min < Rs_ext < self.config.Rs_max):
                stats_counter['constraint_fail'] += 1
                continue
            if r2_a < self.config.min_r_squared:
                stats_counter['constraint_fail'] += 1
                continue
            
            # 檢查 4: Cheung II
            H_I = V_win - (n_ext * V_T * np.log(I_win / constant_term))
            slope_b, intercept_b, r_value_b, _, _ = stats.linregress(I_win, H_I)
            r2_b = r_value_b**2
            
            if r2_b < self.config.min_r_squared * 0.95:
                stats_counter['r2_final_fail'] += 1
                continue
            
            phi_b = intercept_b / n_ext
            Rs_b = slope_b
            
            # 檢查 5: Rs 一致性
            if Rs_ext > 1e-10 and abs(Rs_ext - Rs_b) / Rs_ext > 0.3:
                stats_counter['rs_mismatch'] += 1
                continue
            
            # ✅ 所有檢查通過
            stats_counter['success'] += 1
            score = r2_a * 0.6 + np.log10(I_win[-1]/(I_win[0]+1e-20))/2 * 0.3 + 0.1
            
            if score > best_result.score:
                best_result = FittingResult(
                    success=True,
                    n=n_ext, Rs=Rs_ext, phi_b=phi_b, r_squared=r2_a,
                    fit_range=(V_win[0], V_win[-1]),
                    fit_indices=(start_idx, end_idx),
                    score=score
                )
                self._log(f"  ✓ New best at idx {start_idx}-{end_idx}: n={n_ext:.2f}, ΦB={phi_b:.3f}, R²={r2_a:.4f}")
        
        # === 掃描結束：生成報告 ===
        self._log(f"\n📊 Scan complete. Statistics:")
        self._log(f"  Total windows scanned: {stats_counter['total']}")
        self._log(f"  Monotonic fail: {stats_counter['monotonic_fail']}")
        self._log(f"  R² pre-filter fail: {stats_counter['r2_pre_fail']}")
        self._log(f"  Constraint fail: {stats_counter['constraint_fail']}")
        self._log(f"  R² final fail: {stats_counter['r2_final_fail']}")
        self._log(f"  Rs mismatch: {stats_counter['rs_mismatch']}")
        self._log(f"  ✅ Passed all checks: {stats_counter['success']}")
        
        if n_samples:
            self._log(f"\n📈 n value samples (first 20):")
            self._log(f"  Range: [{min(n_samples):.2f}, {max(n_samples):.2f}]")
            self._log(f"  Mean: {np.mean(n_samples):.2f}, Std: {np.std(n_samples):.2f}")
            self._log(f"  Constraint: [{self.config.n_min}, {self.config.n_max}]")
        
        # === 生成最終警告 ===
        if not best_result.success:
            if stats_counter['total'] == 0:
                warning_msg = f"No windows scanned: data length {len(bias_proc)} < window {window_size}"
            elif stats_counter['monotonic_fail'] == stats_counter['total']:
                warning_msg = "All windows failed monotonicity check: data may be noisy or contain reverse bias"
            elif stats_counter['r2_pre_fail'] == stats_counter['total']:
                warning_msg = f"All windows had R² < 0.9: linear relation poor, may not be thermionic emission"
            elif stats_counter['constraint_fail'] == stats_counter['total']:
                warning_msg = f"All windows failed constraints: n range [{min(n_samples):.2f}, {max(n_samples):.2f}] vs constraint [{self.config.n_min}, {self.config.n_max}]"
            elif stats_counter['success'] > 0:
                warning_msg = f"Found {stats_counter['success']} valid windows but score too low"
            else:
                warning_msg = f"Multi-stage filtering failed. Stats: {stats_counter}"
            
            self._log(f"\n❌ Final warning: {warning_msg}")
            return FittingResult(success=False, warning=warning_msg, diagnostic={'stats': stats_counter, 'n_samples': n_samples[:50]})
        
        self._log(f"\n✅ Fitting successful!")
        return best_result