# src/ivs_tools/config_loader.py
from pathlib import Path
import yaml
from typing import Optional, Dict
import numpy as np
import os


class ConfigLoader:
    """配置載入器：支援巢狀 YAML 結構"""
    
    _cache: Dict[str, Dict] = {}
    
    @classmethod
    def _find_config_dir(cls) -> Path:
        """智能尋找 config 目錄"""
        # 1. 環境變數
        env_config = os.getenv('IVS_TOOLS_CONFIG')
        if env_config and Path(env_config).exists():
            return Path(env_config)
        
        # 2. 當前工作目錄
        cwd_config = Path.cwd() / "config"
        if cwd_config.exists() and any(cwd_config.glob("*.y*ml")):
            return cwd_config
        
        # 3. 專案根目錄
        module_path = Path(__file__).resolve()
        project_root = module_path.parent.parent.parent
        project_config = project_root / "config"
        if project_config.exists() and any(project_config.glob("*.y*ml")):
            return project_config
        
        return Path.cwd() / "config"
    
    @classmethod
    def load(cls, name: str = 'analysis_config', reload: bool = False) -> Dict:
        """載入 YAML 配置 → 返回 Python dict"""
        key = name
        if key in cls._cache and not reload:
            return cls._cache[key]
        
        config_dir = cls._find_config_dir()
        for ext in [".yml", ".yaml"]:
            path = config_dir / f"{name}{ext}"
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
                cls._cache[key] = config
                return config
        
        # YAML 不存在時返回空 dict
        return {}
    
    @classmethod
    def get_area(cls, key: str, default: Optional[float] = None) -> float:
        """獲取接觸面積 (cm²)"""
        config = cls.load()
        areas = config.get('contact_areas', {})
        
        if key not in areas:
            if default is not None:
                return default
            # 硬編碼後備
            defaults = {
                'cafm_tip_10nm': 7.854e-13,
                'cafm_tip_20nm': 3.142e-12,
                'MoS2_monolayer': 7.854e-13,
            }
            if key in defaults:
                return defaults[key]
            raise KeyError(f"Area key '{key}' not found. Available: {list(areas.keys())}")
        
        area = areas[key]
        if area is None:
            raise ValueError(f"Area '{key}' is null")
        return float(area)
    
    @classmethod
    def calc_area_from_diameter(cls, diameter_nm: float) -> float:
        """從 tip 直徑 (nm) 計算面積 (cm²)"""
        r_cm = (diameter_nm / 2) * 1e-7
        return np.pi * r_cm ** 2
    
    # ✅ 新增：獲取物理常數
    @classmethod
    def get_physics_params(cls) -> Dict:
        """
        獲取物理常數（從 YAML 的 experiment + physics 區塊）
        
        Returns
        -------
        dict : {
            'temperature': float,  # K
            'a_star': float,       # A/cm²/K²
            'material': str,
        }
        """
        config = cls.load()
        
        experiment = config.get('experiment', {})
        physics = config.get('physics', {})
        
        return {
            'temperature': experiment.get('temperature_K', 300.0),
            'a_star': physics.get('richardson_constant', 54.0),
            'material': experiment.get('material', 'MoS2'),
        }
    
    # ✅ 新增：獲取擬合參數
    @classmethod
    def get_fitting_config(cls, layer_type: str = "multilayer") -> Dict:
        """
        獲取擬合參數（從 YAML 的 fitting 區塊，支援層數覆蓋）
        
        Returns
        -------
        dict : 擬合參數字典（所有數值已轉換為正確類型）
        """
        config = cls.load()
        fitting = config.get('fitting', {})
        layer_overrides = fitting.get('layer_overrides', {})
        
        # 基礎參數（明確轉換類型）
        result = {
            'window_size': int(fitting.get('window_size', 25)),
            'step_size': int(fitting.get('step_size', 3)),
            'min_r_squared': float(fitting.get('min_r_squared', 0.90)),
            'n_min': float(fitting.get('ideality_factor', {}).get('min', 0.5)),
            'n_max': float(fitting.get('ideality_factor', {}).get('max', 5.0)),
            'Rs_min': float(fitting.get('series_resistance', {}).get('min', 0.0)),
            'Rs_max': float(fitting.get('series_resistance', {}).get('max', 1e10)),
            'savgol_window': int(fitting.get('savgol_window', 31)),
            'savgol_polyorder': int(fitting.get('savgol_polyorder', 3)),
            'current_threshold': float(fitting.get('current_threshold', 5e-13)),
        }
        
        # 層數覆蓋
        if layer_type in layer_overrides:
            override = layer_overrides[layer_type]
            if 'window_size' in override:
                result['window_size'] = int(override['window_size'])
            if 'min_r_squared' in override:
                result['min_r_squared'] = float(override['min_r_squared'])
            if 'ideality_factor' in override:
                if 'max' in override['ideality_factor']:
                    result['n_max'] = float(override['ideality_factor']['max'])
                if 'min' in override['ideality_factor']:
                    result['n_min'] = float(override['ideality_factor']['min'])
        
        return result
    
    # ✅ 新增：獲取批量處理參數
    @classmethod
    def get_batch_params(cls) -> Dict:
        """獲取批量處理參數"""
        config = cls.load()
        batch = config.get('batch', {})
        return {
            'skip_initial_points': batch.get('skip_initial_points', 3),
            'use_parallel': batch.get('use_parallel', False),
            'n_workers': batch.get('n_workers', 8),
        }