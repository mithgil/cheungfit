# src/ivs_tools/physics/schottky_analyzer.py
"""
SchottkyAnalyzer: High-level interface for Schottky barrier analysis
整合配置管理、批量處理、結果格式化
"""
import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path

from .sliding_window_fitter import (
    SlidingWindowCheungFitter, 
    FittingConfig, 
    FittingResult
)


class SchottkyAnalyzer:
    """
    Schottky Barrier Analysis Interface
    
    使用方式:
    --------
    analyzer = SchottkyAnalyzer(area_cm2=1e-12, temperature=300)
    result = analyzer.fit_pixel(bias, current, layer_type="monolayer")
    
    或批量:
    results = analyzer.batch_fit(h5map, pixel_list=[(0,0), (1,1), ...])
    """
    
    def __init__(self, area_cm2: float, temperature: float = 300.0, a_star: float = 54.0):
        """
        Parameters
        ----------
        area_cm2 : float
            有效接觸面積 (cm²)
        temperature : float
            絕對溫度 (K), default=300
        a_star : float
            Richardson constant (A/cm²/K²), MoS₂ ~54
        """
        self.area_cm2 = area_cm2
        self.T = temperature
        self.A_star = a_star
        
        # 物理常數
        self.k_B_eV = 8.617333e-5  # eV/K
        self.k_B_J = 1.380649e-23  # J/K
        self.q = 1.6021766e-19     # C
        
        # 熱電壓
        self.V_T = (self.k_B_J * self.T) / self.q  # ~0.0259 V @ 300K
        
        # 初始化擬合器
        self.fitter = SlidingWindowCheungFitter()
    
    def calculate_barrier(self, slope: float, intercept: float) -> Dict[str, float]:
        """
        從 Cheung Function II 的擬合結果計算 ΦB
        
        Parameters
        ----------
        slope : float
            H(I) vs I 的斜率 (應等於 Rs)
        intercept : float  
            H(I) vs I 的截距 (應等於 n·ΦB)
        n : float
            理想因子 (從 Cheung I 獲得)
            
        Returns
        -------
        dict : {'phi_b': float, 'Rs': float}
        """
        # 注意: 此方法通常由內部自動呼叫，使用者一般不需要直接調用
        # 保留此方法供進階使用者或驗證用
        return {'Rs': slope, 'n_phiB': intercept}
    
    def fit_pixel(self, bias: np.ndarray, current: np.ndarray, 
                  layer_type: str = "multilayer") -> FittingResult:
        """
        對單一像素執行擬合
        
        Parameters
        ----------
        bias : np.ndarray
            偏壓向量 (V)
        current : np.ndarray
            電流向量 (A)
        layer_type : str
            "monolayer" | "bilayer" | "multilayer"
            
        Returns
        -------
        FittingResult
        """
        return self.fitter.fit(
            bias=bias,
            current=current,
            area_cm2=self.area_cm2,
            T=self.T,
            a_star=self.A_star,
            layer_type=layer_type
        )
    
    def batch_fit(self, 
                  h5map,  # HDF5Analyzer instance
                  pixel_list: Optional[List[Tuple[int, int]]] = None,
                  layer_type: str = "multilayer",
                  skip_points: int = 3) -> Dict[Tuple[int, int], FittingResult]:
        """
        批量擬合多個像素
        
        Parameters
        ----------
        h5map : HDF5Analyzer
            已初始化的 HDF5 分析器
        pixel_list : list of (x, y), optional
            要處理的像素座標，None 表示全部
        layer_type : str
            預設層數類型
        skip_points : int
            跳過 I-V 曲線前幾個點 (噪聲)
            
        Returns
        -------
        dict : {(x, y): FittingResult}
        """
        from tqdm import tqdm
        
        metadata = h5map.metadata
        if pixel_list is None:
            pixel_list = [(x, y) for y in range(metadata.y_pixels) 
                         for x in range(metadata.x_pixels)]
        
        results = {}
        
        for x, y in tqdm(pixel_list, desc="Fitting pixels"):
            # 獲取數據
            spectrum = h5map.get_spectrum_at_pixel(x, y)[skip_points:]
            bias = h5map.bias_vector[skip_points:]
            
            # 擬合
            result = self.fit_pixel(bias, spectrum, layer_type=layer_type)
            results[(x, y)] = result
        
        return results
    
    def results_to_dataframe(self, results: Dict) -> 'pd.DataFrame':
        """
        將擬合結果轉為 pandas DataFrame 方便分析
        
        Parameters
        ----------
        results : dict
            batch_fit 的輸出
            
        Returns
        -------
        pd.DataFrame
        """
        import pandas as pd
        
        rows = []
        for (x, y), res in results.items():
            row = {
                'x': x, 'y': y,
                'success': res.success,
                'n': res.n,
                'Rs': res.Rs,
                'phi_b': res.phi_b,
                'r_squared': res.r_squared,
                'score': res.score,
                'fit_range': res.fit_range,
                'warning': res.warning
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def export_heatmap_data(self, results: Dict, 
                           output_path: Union[str, Path]) -> Path:
        """
        導出 2D heatmap 數據 (CSV + NPZ)
        
        Parameters
        ----------
        results : dict
            batch_fit 的輸出
        output_path : str or Path
            輸出路徑 (不含副檔名)
            
        Returns
        -------
        Path : 實際輸出的 .npz 路徑
        """
        import pandas as pd
        output_path = Path(output_path)
        
        # 1. CSV: 表格格式
        df = self.results_to_dataframe(results)
        df.to_csv(f"{output_path}.csv", index=False)
        
        # 2. NPZ: 2D array 格式 (方便 imshow)
        if results:
            # 從任意 result 獲取 metadata
            sample_res = next(iter(results.values()))
            # 這裡假設 h5map metadata 可訪問，實際使用時需傳入
            # 簡化版：只儲存成功結果的 2D map
            success_map = {
                (x, y): res.phi_b if res.success else np.nan 
                for (x, y), res in results.items()
            }
            # 轉為 2D array 需要知道 dimensions，建議由 caller 處理
            np.savez(f"{output_path}.npz", 
                     results=results,
                     timestamp=np.datetime64('now'))
        
        return output_path.with_suffix('.npz')