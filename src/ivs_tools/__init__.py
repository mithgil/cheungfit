# src/ivs_tools/__init__.py
"""
ivs_tools: Nanonis SPM Data Processing and Schottky Analysis
"""
__version__ = "0.1.0"
__author__ = "Tim"

# ── 快捷進口 (方便 notebook 使用) ──
# IO 模組
from .io.h5utils import HDF5Analyzer

# Physics 模組 (新流程核心)
from .physics.sliding_window_fitter import (
    SlidingWindowCheungFitter,
    FittingConfig,
    FittingResult
)

# 可選：保留舊函數供過渡期使用
# from .physics.cheung_analysis import cheung_analysis

# Viz 模組
# from .viz.cheung_diagnostic import plot_fitting_diagnostic

# ── 控制 from ivs_tools import * 的行為 ──
__all__ = [
    "__version__",
    "HDF5Analyzer",
    "SlidingWindowCheungFitter",
    "FittingConfig", 
    "FittingResult",
    # "cheung_analysis",  # 若 archive 則移除
]

# ── 延遲載入 (可選，減少 import 時間) ──
# 如果模組載入較重，可用 __getattr__ 動態匯入
# def __getattr__(name):
#     if name == "plot_fitting_diagnostic":
#         from .viz.cheung_diagnostic import plot_fitting_diagnostic
#         return plot_fitting_diagnostic
#     raise AttributeError(f"module {__name__!r} has no attribute {name!r}")