# src/ivs_tools/config_loader.py
from pathlib import Path
import yaml
from typing import Any, Dict, Optional

class ConfigLoader:
    """統一配置管理：避免硬編碼，支援熱更新"""
    
    _cache: Dict[str, Dict] = {}
    _base_path = Path(__file__).parent.parent / "config"
    
    @classmethod
    def load(cls, name: str, reload: bool = False) -> Dict:
        """載入 config/{name}.yml"""
        key = f"{name}"
        if key in cls._cache and not reload:
            return cls._cache[key]
        
        # 嘗試 .yml 或 .yaml
        for ext in [".yml", ".yaml"]:
            path = cls._base_path / f"{name}{ext}"
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                cls._cache[key] = config or {}
                return cls._cache[key]
        
        raise FileNotFoundError(f"Config '{name}' not found in {cls._base_path}")
    
    @classmethod
    def get_area(cls, key: str, default: Optional[float] = None) -> float:
        """便捷獲取接觸面積"""
        areas = cls.load('area_settings')
        area = areas.get('contact_areas', {}).get(key)
        if area is None and default is not None:
            return default
        if area is None:
            raise KeyError(f"Area key '{key}' not found. Available: {list(areas.get('contact_areas', {}).keys())}")
        return area
    
    @classmethod
    def get_fitting_config(cls, layer_type: Optional[str] = None) -> Dict:
        """獲取擬合參數，支援層數覆蓋"""
        cfg = cls.load('default').get('fitting', {})
        if layer_type == "monolayer":
            # 單層通常線性區較短，調整窗口
            cfg = {**cfg, "window_size": 30, "min_r_squared": 0.99}
        return cfg