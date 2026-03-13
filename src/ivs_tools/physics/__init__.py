# src/ivs_tools/physics/__init__.py
"""
Physics module: Schottky barrier analysis tools
"""

# ✅ 核心算法
from .sliding_window_fitter import (
    SlidingWindowCheungFitter,
    FittingConfig,
    FittingResult
)

# ✅ 高層接口
from .schottky_analyzer import SchottkyAnalyzer  # ✅ 加上這行！

# ❌ 舊模組 - 保持註解
# from .archive.cheung_analysis import cheung_analysis

__all__ = [
    "SlidingWindowCheungFitter",
    "FittingConfig", 
    "FittingResult",
    "SchottkyAnalyzer"  # ✅ 加上這行！
    # "cheung_analysis",
]