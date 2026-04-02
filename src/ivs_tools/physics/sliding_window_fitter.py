# src/ivs_tools/physics/sliding_window_fitter.py
"""
Sliding Window Cheung Method Fitter
核心算法模組：獨立、可測試、完整診斷輸出
"""
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
import warnings

# src/ivs_tools/physics/sliding_window_fitter.py

@dataclass
class FittingConfig:
    """擬合參數配置 - 支援 YAML 載入與手動覆蓋"""
    # 窗口參數
    window_size: int = 25
    step_size: int = 3
    min_r_squared: float = 0.90
    
    # 物理約束
    n_min: float = 0.5
    n_max: float = 5.0
    Rs_min: float = 0.0
    Rs_max: float = 1e10
    
    # 數據預處理
    savgol_window: int = 31
    savgol_polyorder: int = 3
    current_threshold: float = 5e-13
    
    # 層數相關
    layer_type: str = "multilayer"
    
    def __post_init__(self):
        """
        數據類初始化後自動執行：確保類型正確
        解決 YAML 載入時數字變字串的問題
        """
        # 強制轉換為正確類型
        self.window_size = int(self.window_size)
        self.step_size = int(self.step_size)
        self.min_r_squared = float(self.min_r_squared)
        self.n_min = float(self.n_min)
        self.n_max = float(self.n_max)
        self.Rs_min = float(self.Rs_min)
        self.Rs_max = float(self.Rs_max)
        self.savgol_window = int(self.savgol_window)
        self.savgol_polyorder = int(self.savgol_polyorder)
        self.current_threshold = float(self.current_threshold)
        self.layer_type = str(self.layer_type)
    
    def update_for_layer(self, layer_type: str):
        """根據層數動態調整參數"""
        if layer_type == "monolayer":
            self.window_size = 25
            self.n_max = 5.0
            self.min_r_squared = 0.90
        elif layer_type == "bilayer":
            self.window_size = 30
            self.n_max = 4.0
        elif layer_type == "multilayer":
            self.window_size = 25
            self.n_max = 5.0
            self.min_r_squared = 0.90

@dataclass
class FittingResult:
    """單次擬合結果"""
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

class SlidingWindowCheungFitter:
    """
    滑動窗口 Cheung method 擬合器
    
    輸入：bias (V), current (A), area_cm2
    輸出：FittingResult (n, Rs, phi_b, diagnostic)
    """
    
    def __init__(self, config: Optional[FittingConfig] = None):
        self.config = config or FittingConfig()
        self.q = 1.6021766e-19
        self.last_diagnostic = None
        
    def _preprocess(self, bias: np.ndarray, current: np.ndarray, 
                   forward_bias_only: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        預處理：濾波 + 去除 noise floor + 選擇正向偏壓
        
        Parameters
        ----------
        bias : np.ndarray
            偏壓向量 (V)
        current : np.ndarray
            電流向量 (A)
        forward_bias_only : bool
            是否只保留正向偏壓 (V > 0)
            
        Returns
        -------
        tuple : (bias_valid, I_smooth)
        """
        # ── 步驟 1: 取絕對值 + 電流閾值篩選 ──
        I_abs = np.abs(current)
        mask_current = I_abs > self.config.current_threshold
        
        # ── 步驟 2: 正向偏壓篩選 ──
        if forward_bias_only:
            mask_forward = bias > 0
        else:
            mask_forward = np.ones_like(bias, dtype=bool)
        
        # ── 步驟 3: 合併遮罩 ──
        mask_valid = mask_current & mask_forward
        
        if np.sum(mask_valid) < 10:
            raise ValueError(
                f"有效數據點不足：{np.sum(mask_valid)} < 10\n"
                f"   總點數：{len(bias)}\n"
                f"   電流閾值：{self.config.current_threshold:.2e} A\n"
                f"   正向偏壓點：{np.sum(bias > 0)}\n"
                f"   建議：降低 current_threshold 或關閉 forward_bias_only"
            )
        
        bias_valid = bias[mask_valid]
        I_valid = I_abs[mask_valid]
        
        # ── 步驟 4: Savitzky-Golay 平滑 ──
        window = min(self.config.savgol_window, len(I_valid) - 2)
        if window % 2 == 0: window += 1
        
        I_smooth = savgol_filter(I_valid, window_length=window, 
                                polyorder=self.config.savgol_polyorder)
        
        return bias_valid, I_smooth
    
    def _compute_cheung_I(self, bias: np.ndarray, I: np.ndarray, T: float) -> Tuple[np.ndarray, float]:
        """計算 Cheung Function I: dV/d(lnI)"""
        k_B_J = 1.380649e-23
        V_T = (k_B_J * T) / self.q
        
        ln_I = np.log(I + 1e-20)
        dV_dlnI = np.gradient(bias, ln_I)
        return dV_dlnI, V_T
    
    def _linear_fit(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        """線性擬合：返回 (slope, intercept, r_squared)"""
        if len(x) < 5:
            return 0, 0, -1
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        return slope, intercept, r_value**2
    
    def _check_constraints(self, n: float, Rs: float, r2: float) -> Tuple[bool, Optional[str]]:
        """物理約束檢查"""
        if r2 < self.config.min_r_squared:
            return False, f"R²={r2:.3f} < {self.config.min_r_squared}"
        if not (self.config.n_min < n < self.config.n_max):
            return False, f"n={n:.2f} outside [{self.config.n_min}, {self.config.n_max}]"
        if not (self.config.Rs_min < Rs < self.config.Rs_max):
            return False, f"Rs={Rs:.2e} out of range"
        return True, None
    
    def _compute_score(self, r2: float, I_range: Tuple[float, float], residuals: np.ndarray) -> float:
        """綜合評分"""
        score_r2 = r2 * 0.6
        log_span = np.log10(I_range[1] / (I_range[0] + 1e-20))
        score_range = min(log_span / 2.0, 1.0) * 0.3
        smoothness = 1.0 / (1.0 + np.std(residuals) * 1e3) if len(residuals) > 0 else 0
        score_smooth = smoothness * 0.1
        return score_r2 + score_range + score_smooth
    
    def fit(self, bias: np.ndarray, current: np.ndarray, 
            area_cm2: float, T: float = 300, a_star: float = 54.0,
            layer_type: Optional[str] = None,
            forward_bias_only: bool = True,
            min_forward_bias: float = 0.0) -> FittingResult:
        """
        執行滑動窗口擬合
        
        Parameters
        ----------
        bias : np.ndarray
            偏壓向量 (V)
        current : np.ndarray
            電流向量 (A)
        area_cm2 : float
            有效接觸面積 (cm²)
        T : float
            溫度 (K)
        a_star : float
            Richardson constant (A/cm²/K²)
        layer_type : str, optional
            "monolayer" | "bilayer" | "multilayer"
        forward_bias_only : bool, default=True
            是否只處理正向偏壓 (V > 0)
        min_forward_bias : float, default=0.0
            正向偏壓最小值 (V)，例如 0.1 表示只分析 V > 0.1 V
            
        Returns
        -------
        FittingResult
        """
        
        if layer_type:
            self.config.update_for_layer(layer_type)
        
        # === 診斷計數器 ===
        stats_counter = {
            'total': 0, 'monotonic_fail': 0, 'r2_pre_fail': 0, 
            'constraint_fail': 0, 'r2_final_fail': 0, 'rs_mismatch': 0, 'success': 0
        }
        n_samples = []
        r2_samples = []
        n_vs_V = []
        r2_vs_V = []
        valid_windows = []
        
        try:
            # 預處理時篩選正向偏壓
            bias_proc, I_proc = self._preprocess(bias, current, forward_bias_only=forward_bias_only)
        except ValueError as e:
            return FittingResult(success=False, warning=f"預處理失敗：{str(e)}")
        
        # 額外篩選：min_forward_bias
        if min_forward_bias > 0:
            mask_min_bias = bias_proc > min_forward_bias
            bias_proc = bias_proc[mask_min_bias]
            I_proc = I_proc[mask_min_bias]
        
        if len(bias_proc) < self.config.window_size:
            return FittingResult(
                success=False, 
                warning=f"數據點不足：{len(bias_proc)} < {self.config.window_size}",
                diagnostic={
                    'data_points': len(bias_proc), 
                    'required': self.config.window_size,
                    'bias_range': (bias_proc.min() if len(bias_proc) > 0 else None,
                                  bias_proc.max() if len(bias_proc) > 0 else None)
                }
            )
        
        dV_dlnI, V_T = self._compute_cheung_I(bias_proc, I_proc, T)
        constant_term = area_cm2 * a_star * (T**2)
        
        best_result = FittingResult(success=False)
        best_score = 0
        
        # === 滑動窗口掃描 ===
        window = self.config.window_size
        step = self.config.step_size
        
        for start_idx in range(0, len(bias_proc) - window + 1, step):
            end_idx = start_idx + window
            stats_counter['total'] += 1
            
            I_win = I_proc[start_idx:end_idx]
            V_win = bias_proc[start_idx:end_idx]
            y_win = dV_dlnI[start_idx:end_idx]
            
            # 檢查 1: 電流單調性
            if not np.all(np.diff(I_win) > 0):
                stats_counter['monotonic_fail'] += 1
                continue
            
            # 檢查 2: Cheung I 線性擬合
            slope_a, intercept_a, r2_a = self._linear_fit(I_win, y_win)
            n_ext = intercept_a / V_T
            Rs_ext = slope_a
            
            # 收集診斷數據
            n_samples.append(n_ext)
            r2_samples.append(r2_a)
            V_center = (V_win[0] + V_win[-1]) / 2
            n_vs_V.append((V_center, n_ext))
            r2_vs_V.append((V_center, r2_a))
            
            if r2_a < 0.8:  # 預篩
                stats_counter['r2_pre_fail'] += 1
                continue
            
            # 檢查 3: 物理約束
            passed, warn = self._check_constraints(n_ext, Rs_ext, r2_a)
            if not passed:
                stats_counter['constraint_fail'] += 1
                continue
            
            # 檢查 4: Cheung II
            H_I = V_win - (n_ext * V_T * np.log(I_win / constant_term))
            slope_b, intercept_b, r2_b = self._linear_fit(I_win, H_I)
            
            if r2_b < self.config.min_r_squared * 0.95:
                stats_counter['r2_final_fail'] += 1
                continue
                
            phi_b = intercept_b / n_ext
            Rs_b = slope_b
            
            # 檢查 5: Rs 交叉驗證
            if Rs_ext > 1e-10 and abs(Rs_ext - Rs_b) / Rs_ext > 0.3:
                stats_counter['rs_mismatch'] += 1
                continue
            
            # ✅ 所有檢查通過
            stats_counter['success'] += 1
            valid_windows.append({
                'n': n_ext, 'phi_b': phi_b, 'r2': r2_a,
                'V_range': (V_win[0], V_win[-1]),
                'indices': (start_idx, end_idx)
            })
            
            # 計算評分
            residuals = y_win - (slope_a * I_win + intercept_a)
            score = self._compute_score(r2_a, (I_win[0], I_win[-1]), residuals)
            
            if score > best_score:
                best_score = score
                best_result = FittingResult(
                    success=True,
                    n=n_ext,
                    Rs=Rs_ext,
                    phi_b=phi_b,
                    r_squared=r2_a,
                    fit_range=(V_win[0], V_win[-1]),
                    fit_indices=(start_idx, end_idx),
                    score=score,
                    diagnostic={
                        'bias': bias_proc, 'I': I_proc, 'dV_dlnI': dV_dlnI,
                        'window_I': I_win, 'window_y': y_win,
                        'slope_a': slope_a, 'intercept_a': intercept_a,
                        'slope_b': slope_b, 'intercept_b': intercept_b,
                        'n_samples': n_samples, 'r2_samples': r2_samples,
                        'n_vs_V': n_vs_V, 'r2_vs_V': r2_vs_V,
                        'stats_counter': stats_counter,
                        'valid_windows': valid_windows,
                        'preprocess_info': {
                            'original_points': len(bias),
                            'forward_bias_points': np.sum(bias > 0),
                            'after_threshold': len(bias_proc),
                            'forward_bias_only': forward_bias_only,
                            'min_forward_bias': min_forward_bias
                        }
                    }
                )
        
        # === 擬合失敗：生成警告 ===
        if not best_result.success:
            if stats_counter['total'] == 0:
                warning_msg = f"無可掃描窗口：數據點 {len(bias_proc)} < 窗口 {window}"
            elif stats_counter['monotonic_fail'] == stats_counter['total']:
                warning_msg = "所有窗口電流非單調：可能數據噪聲大"
            elif stats_counter['r2_pre_fail'] == stats_counter['total']:
                warning_msg = "所有窗口 R² < 0.8：線性關係差"
            elif stats_counter['constraint_fail'] == stats_counter['total']:
                warning_msg = f"所有窗口 n 或 R² 超出約束：n 範圍 [{min(n_samples):.2f}, {max(n_samples):.2f}]"
            else:
                warning_msg = f"多階段篩選失敗。通過：{stats_counter['success']}/{stats_counter['total']}"
            
            best_result = FittingResult(
                success=False,
                warning=warning_msg,
                diagnostic={
                    'stats_counter': stats_counter,
                    'n_samples': n_samples, 'r2_samples': r2_samples,
                    'n_vs_V': n_vs_V, 'r2_vs_V': r2_vs_V,
                    'bias': bias_proc, 'I': I_proc, 'dV_dlnI': dV_dlnI,
                    'preprocess_info': {
                        'original_points': len(bias),
                        'forward_bias_points': np.sum(bias > 0),
                        'after_threshold': len(bias_proc),
                        'forward_bias_only': forward_bias_only
                    }
                }
            )
        
        self.last_diagnostic = best_result.diagnostic
        return best_result
    
    def get_diagnostic_data(self) -> Optional[Dict]:
        """返回診斷數據供可視化"""
        return self.last_diagnostic.copy() if self.last_diagnostic else None
    