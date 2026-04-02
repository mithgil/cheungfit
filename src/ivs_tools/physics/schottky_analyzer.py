# src/ivs_tools/physics/schottky_analyzer.py
"""
SchottkyAnalyzer: High-level interface for Schottky barrier analysis
整合 YAML 配置、批量處理、結果格式化
"""
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path

from .sliding_window_fitter import (
    SlidingWindowCheungFitter, 
    FittingConfig, 
    FittingResult
)
from ..config_loader import ConfigLoader


class SchottkyAnalyzer:
    """
    Schottky Barrier Analysis Interface
    
    參數優先級：
    1. __init__ 顯式傳入 (最高)
    2. YAML 配置預設
    3. 硬編碼預設 (最低)
    
    使用方式:
    --------
    # 方式 1: 完全從 YAML 載入
    analyzer = SchottkyAnalyzer()
    
    # 方式 2: 只覆蓋面積
    analyzer = SchottkyAnalyzer(area_cm2=1e-12)
    
    # 方式 3: 完全自訂
    analyzer = SchottkyAnalyzer(
        area_cm2=1e-12, temperature=350, a_star=60, load_from_yaml=False
    )
    """
    
    def __init__(
        self, 
        area_cm2: Optional[float] = None,
        temperature: Optional[float] = None,
        a_star: Optional[float] = None,
        layer_type: str = "multilayer",
        load_from_yaml: bool = True,
        fitting_config: Optional[FittingConfig] = None
    ):
        """
        Parameters
        ----------
        area_cm2 : float, optional
            有效接觸面積 (cm²)
        temperature : float, optional
            溫度 (K)
        a_star : float, optional
            Richardson constant (A/cm²/K²)
        layer_type : str
            層數類型
        load_from_yaml : bool
            是否從 YAML 載入預設
        fitting_config : FittingConfig, optional
            自訂擬合參數
        """
        # 載入 YAML 預設
        if load_from_yaml:
            try:
                physics = ConfigLoader.get_physics_params()
                default_area = ConfigLoader.get_area('cafm_tip_10nm')
                if fitting_config is None:
                    fitting_cfg_dict = ConfigLoader.get_fitting_config(layer_type)
                    fitting_config = FittingConfig(**fitting_cfg_dict)
            except (FileNotFoundError, KeyError):
                physics = {'temperature': 300.0, 'a_star': 54.0}
                default_area = 7.854e-13
        else:
            physics = {'temperature': 300.0, 'a_star': 54.0}
            default_area = 7.854e-13
        
        # 參數優先級：傳入 > YAML > 硬編碼
        self.area_cm2 = area_cm2 if area_cm2 is not None else default_area
        self.T = temperature if temperature is not None else physics['temperature']
        self.A_star = a_star if a_star is not None else physics['a_star']
        self.layer_type = layer_type
        
        # 物理常數
        self.k_B_J = 1.380649e-23
        self.q = 1.6021766e-19
        self.V_T = (self.k_B_J * self.T) / self.q
        
        # 初始化擬合器
        self.fitter = SlidingWindowCheungFitter(config=fitting_config)
    
    def fit_pixel(self, bias: np.ndarray, current: np.ndarray, 
                  layer_type: Optional[str] = None,
                  forward_bias_only: bool = True,
                  min_forward_bias: float = 0.0) -> FittingResult:
        """
        對單一像素執行擬合
        
        Parameters
        ----------
        bias : np.ndarray
            偏壓向量 (V)
        current : np.ndarray
            電流向量 (A)
        layer_type : str, optional
            層數類型
        forward_bias_only : bool, default=True
            是否只處理正向偏壓
        min_forward_bias : float, default=0.0
            正向偏壓最小值 (V)
        """
        layer = layer_type or self.layer_type
        return self.fitter.fit(
            bias=bias,
            current=current,
            area_cm2=self.area_cm2,
            T=self.T,
            a_star=self.A_star,
            layer_type=layer,
            forward_bias_only=forward_bias_only,
            min_forward_bias=min_forward_bias
        )
    
    def batch_fit(self, h5map, pixel_list: Optional[List[Tuple[int, int]]] = None,
                  layer_type: Optional[str] = None, skip_points: Optional[int] = None,
                  forward_bias_only: bool = True,
                  min_forward_bias: float = 0.1) -> Dict:  # ← 新增參數
        """
        批量擬合多個像素
        
        Parameters
        ----------
        forward_bias_only : bool, default=True
            是否只處理正向偏壓
        min_forward_bias : float, default=0.1
            最小正向偏壓 (V)
        """
        from tqdm import tqdm
        
        if skip_points is None:
            try:
                config = ConfigLoader.load()
                skip_points = config.get('batch', {}).get('skip_initial_points', 3)
            except:
                skip_points = 3
        
        metadata = h5map.metadata
        if pixel_list is None:
            pixel_list = [(x, y) for y in range(metadata.y_pixels) 
                         for x in range(metadata.x_pixels)]
        
        results = {}
        for x, y in tqdm(pixel_list, desc="Fitting pixels"):
            spectrum = h5map.get_spectrum_at_pixel(x, y)[skip_points:]
            bias = h5map.bias_vector[skip_points:]
            
            # ✅ 關鍵：傳遞 forward_bias_only 和 min_forward_bias
            result = self.fit_pixel(
                bias, spectrum, 
                layer_type=layer_type,
                forward_bias_only=forward_bias_only,
                min_forward_bias=min_forward_bias
            )
            results[(x, y)] = result
        
        return results

    def fit_pixel_adaptive(self, bias: np.ndarray, current: np.ndarray, 
                          layer_type: Optional[str] = None,
                          forward_bias_only: bool = True,
                          min_forward_bias: float = 0.1,
                          max_attempts: int = 3) -> FittingResult:
        """
        自適應擬合：自動放寬約束直到成功
        
        Parameters
        ----------
        max_attempts : int
            最大嘗試次數
        """
        # 定義多組參數配置（從嚴格到寬鬆）
        config_attempts = [
            # 嘗試 1: 標準參數
            {'n_min': 0.5, 'n_max': 5.0, 'min_r_squared': 0.90, 'window_size': 25},
            # 嘗試 2: 放寬 R²
            {'n_min': 0.5, 'n_max': 6.0, 'min_r_squared': 0.85, 'window_size': 25},
            # 嘗試 3: 更寬鬆
            {'n_min': 0.3, 'n_max': 8.0, 'min_r_squared': 0.80, 'window_size': 20},
        ]
        
        for i, cfg in enumerate(config_attempts[:max_attempts]):
            # 臨時修改配置
            original_config = self.fitter.config
            from dataclasses import replace
            self.fitter.config = replace(
                self.fitter.config,
                n_min=cfg['n_min'],
                n_max=cfg['n_max'],
                min_r_squared=cfg['min_r_squared'],
                window_size=cfg['window_size']
            )
            
            # 執行擬合
            result = self.fit_pixel(
                bias=bias,
                current=current,
                layer_type=layer_type,
                forward_bias_only=forward_bias_only,
                min_forward_bias=min_forward_bias
            )
            
            if result.success:
                print(f"✅ 嘗試 {i+1} 成功 (n=[{cfg['n_min']},{cfg['n_max']}], R²>{cfg['min_r_squared']})")
                return result
            else:
                print(f"⚠️ 嘗試 {i+1} 失敗：{result.warning[:50]}...")
        
        # 所有嘗試都失敗，恢復原配置
        self.fitter.config = original_config
        return result  # 返回最後一次結果
    
    def batch_fit_adaptive(self, h5map, pixel_list: Optional[List[Tuple[int, int]]] = None,
                          layer_type: Optional[str] = None, **kwargs) -> Dict:
        """批量自適應擬合"""
        from tqdm import tqdm
        
        if pixel_list is None:
            pixel_list = [(x, y) for y in range(h5map.metadata.y_pixels) 
                         for x in range(h5map.metadata.x_pixels)]
        
        results = {}
        for x, y in tqdm(pixel_list, desc="Adaptive fitting"):
            spectrum = h5map.get_spectrum_at_pixel(x, y)[3:]
            bias = h5map.bias_vector[3:]
            result = self.fit_pixel_adaptive(bias, spectrum, layer_type=layer_type, **kwargs)
            results[(x, y)] = result
        
        return results
        
    def results_to_dataframe(self, results: Dict) -> 'pd.DataFrame':
        """轉為 DataFrame"""
        import pandas as pd
        rows = []
        for (x, y), res in results.items():
            row = {
                'x': x, 'y': y,
                'success': res.success,
                'n': res.n, 'Rs': res.Rs, 'phi_b': res.phi_b,
                'r_squared': res.r_squared, 'score': res.score,
                'fit_range': res.fit_range, 'warning': res.warning
            }
            rows.append(row)
        return pd.DataFrame(rows)
    
    def print_config_summary(self):
        """打印配置摘要"""
        print("🔬 SchottkyAnalyzer Configuration")
        print("=" * 50)
        print(f"   Area: {self.area_cm2:.3e} cm²")
        print(f"   Temperature: {self.T} K")
        print(f"   A**: {self.A_star} A/cm²/K²")
        print(f"   Layer type: {self.layer_type}")
        print(f"   V_T: {self.V_T:.4f} V")
        print(f"   Window: {self.fitter.config.window_size} pts")
        print(f"   n range: [{self.fitter.config.n_min}, {self.fitter.config.n_max}]")
        print(f"   min R²: {self.fitter.config.min_r_squared}")