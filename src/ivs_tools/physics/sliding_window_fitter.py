# src/ivs_tools/physics/sliding_window_fitter.py
import numpy as np
from scipy import stats
from scipy.signal import savgol_filter, find_peaks
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List
import warnings

@dataclass
class FittingConfig:
    """擬合參數配置 - 可針對不同層數動態調整"""
    # 窗口參數
    window_size: int = 50
    min_window_size: int = 20
    step_size: int = 5
    
    # 物理約束
    n_min: float = 1.0
    n_max: float = 2.5  # MoS₂ 通常 n < 2.5，若 >3 可能為 Tunneling
    Rs_min: float = 0
    Rs_max: float = 1e10  # 排除數值發散
    min_r_squared: float = 0.98
    
    # 數據預處理
    savgol_window: int = 21
    savgol_polyorder: int = 3
    current_threshold: float = 1e-12  # 忽略低於此值的電流（noise floor）
    
    # 評分權重
    r2_weight: float = 0.6
    range_weight: float = 0.3  # 偏好電流跨度大的區間
    smoothness_weight: float = 0.1
    
    # 層數相關（可外部傳入覆蓋）
    layer_type: str = "multilayer"  # "monolayer" | "bilayer" | "multilayer"
    
    def update_for_layer(self, layer_type: str):
        """根據層數動態調整參數"""
        if layer_type == "monolayer":
            self.window_size = 30
            self.n_max = 2.0
            self.min_r_squared = 0.99
        elif layer_type == "bilayer":
            self.window_size = 40
            self.n_max = 2.2
        # multilayer 使用 default

@dataclass
class FittingResult:
    """單次擬合結果"""
    success: bool
    n: Optional[float] = None
    Rs: Optional[float] = None
    phi_b: Optional[float] = None
    r_squared: Optional[float] = None
    fit_range: Optional[Tuple[float, float]] = None  # (V_start, V_end)
    fit_indices: Optional[Tuple[int, int]] = None
    score: float = 0.0
    warning: Optional[str] = None
    diagnostic: Dict = field(default_factory=dict)  # 儲存中間數據供可視化

class SlidingWindowCheungFitter:
    """
    滑動窗口 Cheung method 擬合器
    支援：動態窗口、物理約束過濾、評分機制、診斷數據輸出
    """
    
    def __init__(self, config: Optional[FittingConfig] = None):
        self.config = config or FittingConfig()
        self.k_B = 8.617333e-5  # eV/K
        self.q = 1.6021766e-19
        self.last_diagnostic = None  # 儲存最後一次擬合的診斷數據
        
    def _preprocess(self, bias: np.ndarray, current: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """預處理：濾波 + 去除 noise floor"""
        # 1. 取絕對值並過濾 noise
        I_abs = np.abs(current)
        mask_valid = I_abs > self.config.current_threshold
        
        if np.sum(mask_valid) < 10:
            raise ValueError("有效數據點不足，請檢查 current_threshold 或數據品質")
        
        bias_valid = bias[mask_valid]
        I_valid = I_abs[mask_valid]
        
        # 2. Savitzky-Golay 平滑
        window = min(self.config.savgol_window, len(I_valid) - 2)
        if window % 2 == 0: window += 1  # 必須為奇數
        
        I_smooth = savgol_filter(I_valid, window_length=window, 
                                polyorder=self.config.savgol_polyorder)
        
        return bias_valid, I_smooth
    
    def _compute_cheung_functions(self, bias: np.ndarray, I: np.ndarray, 
                                  T: float = 300) -> Tuple[np.ndarray, np.ndarray, float]:
        """計算 Cheung 函數 I: dV/d(lnI) 與輔助參數"""
        V_T = (1.380649e-23 * T) / self.q
        
        # 數值微分：dV/d(lnI)
        ln_I = np.log(I + 1e-20)  # 避免 log(0)
        dV_dlnI = np.gradient(bias, ln_I)
        
        return dV_dlnI, V_T
    
    def _linear_fit_window(self, x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        """執行線性擬合，返回 (slope, intercept, r_squared)"""
        if len(x) < 5:
            return 0, 0, -1
        
        slope, intercept, r_value, _, _ = stats.linregress(x, y)
        return slope, intercept, r_value**2
    
    def _check_physical_constraints(self, n: float, Rs: float, r2: float) -> Tuple[bool, Optional[str]]:
        """檢查物理約束，返回 (是否通過, 警告訊息)"""
        if r2 < self.config.min_r_squared:
            return False, f"R²={r2:.3f} < {self.config.min_r_squared}"
        if not (self.config.n_min < n < self.config.n_max):
            return False, f"n={n:.2f} 超出 [{self.config.n_min}, {self.config.n_max}]"
        if not (self.config.Rs_min < Rs < self.config.Rs_max):
            return False, f"Rs={Rs:.2e} 超出合理範圍"
        return True, None
    
    def _compute_score(self, r2: float, current_range: Tuple[float, float], 
                       residuals: np.ndarray) -> float:
        """綜合評分函數"""
        # 1. R² 分數
        score_r2 = r2 * self.config.r2_weight
        
        # 2. 電流跨度分數（偏好更寬的擬合區間）
        log_span = np.log10(current_range[1] / (current_range[0] + 1e-20))
        score_range = min(log_span / 2.0, 1.0) * self.config.range_weight
        
        # 3. 平滑度分數（殘差標準差越小越好）
        if len(residuals) > 0:
            smoothness = 1.0 / (1.0 + np.std(residuals) * 1e3)
        else:
            smoothness = 0
        score_smooth = smoothness * self.config.smoothness_weight
        
        return score_r2 + score_range + score_smooth
    
    def fit(self, bias: np.ndarray, current: np.ndarray, 
            area_cm2: float, T: float = 300, a_star: float = 54.0,
            layer_type: Optional[str] = None) -> FittingResult:
        """
        執行滑動窗口擬合
        
        Parameters:
        -----------
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
            "monolayer" | "bilayer" | "multilayer"，用於動態調整參數
        """
        # 更新層數相關參數
        if layer_type:
            self.config.update_for_layer(layer_type)
        
        try:
            bias_proc, I_proc = self._preprocess(bias, current)
        except ValueError as e:
            return FittingResult(success=False, warning=str(e))
        
        dV_dlnI, V_T = self._compute_cheung_functions(bias_proc, I_proc, T)
        constant_term = area_cm2 * a_star * (T**2)
        
        best_result = FittingResult(success=False)
        n_candidates = []  # 儲存所有候選區間的 n 值供診斷
        
        # === 滑動窗口掃描 ===
        window = self.config.window_size
        step = self.config.step_size
        
        for start_idx in range(0, len(bias_proc) - window + 1, step):
            end_idx = start_idx + window
            
            # 提取窗口數據
            I_win = I_proc[start_idx:end_idx]
            V_win = bias_proc[start_idx:end_idx]
            y_win = dV_dlnI[start_idx:end_idx]
            
            # 快速篩選：電流需單調遞增（正向偏壓區）
            if not np.all(np.diff(I_win) > 0):
                continue
            
            # --- Cheung Method I: dV/d(lnI) vs I ---
            slope_a, intercept_a, r2_a = self._linear_fit_window(I_win, y_win)
            if r2_a < 0.9:  # 快速預篩
                continue
                
            n_extracted = intercept_a / V_T
            Rs_extracted = slope_a
            
            # 物理約束檢查
            passed, warn_msg = self._check_physical_constraints(n_extracted, Rs_extracted, r2_a)
            if not passed:
                n_candidates.append({'n': n_extracted, 'r2': r2_a, 'range': (V_win[0], V_win[-1]), 'reason': warn_msg})
                continue
            
            # --- Cheung Method II: H(I) vs I ---
            H_I = V_win - (n_extracted * V_T * np.log(I_win / constant_term))
            slope_b, intercept_b, r2_b = self._linear_fit_window(I_win, H_I)
            
            if r2_b < self.config.min_r_squared * 0.95:
                continue
                
            phi_b = intercept_b / n_extracted
            Rs_b = slope_b
            
            # 交叉驗證 Rs（兩種方法應接近）
            if abs(Rs_extracted - Rs_b) / (Rs_extracted + 1e-10) > 0.3:
                continue  # Rs 不一致，跳過
            
            # 計算綜合評分
            residuals = y_win - (slope_a * I_win + intercept_a)
            score = self._compute_score(r2_a, (I_win[0], I_win[-1]), residuals)
            
            # 更新最佳結果
            if score > best_result.score:
                best_result = FittingResult(
                    success=True,
                    n=n_extracted,
                    Rs=Rs_extracted,
                    phi_b=phi_b,
                    r_squared=r2_a,
                    fit_range=(V_win[0], V_win[-1]),
                    fit_indices=(start_idx, end_idx),
                    score=score,
                    diagnostic={
                        'bias': bias_proc,
                        'I': I_proc,
                        'dV_dlnI': dV_dlnI,
                        'H_I_full': bias_proc - (n_extracted * V_T * np.log(I_proc / constant_term)),
                        'window_I': I_win,
                        'window_y': y_win,
                        'window_H': H_I,
                        'candidates': n_candidates[-10:]  # 保留最近 10 個被過濾的候選
                    }
                )
        
        self.last_diagnostic = best_result.diagnostic
        return best_result
    
    def get_diagnostic_plot_data(self) -> Optional[Dict]:
        """返回用於可視化診斷的數據"""
        if not self.last_diagnostic:
            return None
        return self.last_diagnostic.copy()