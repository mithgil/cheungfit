# src/ivs_tools/__init__.py
"""
ivs_tools: Nanonis SPM Data Processing and Schottky Analysis
"""
__version__ = "0.1.0"
__author__ = "Tim"

# ── IO 模組 ──
from .io.h5utils import HDF5Analyzer

# ── Config 模組 ──
from .config_loader import ConfigLoader  # ✅ 加上這行

# ── Physics 模組 ──
from .physics.sliding_window_fitter import (
    SlidingWindowCheungFitter,
    FittingConfig,
    FittingResult
)
from .physics.schottky_analyzer import SchottkyAnalyzer

# ── Viz 模組 (可選) ──
# from .viz.cheung_diagnostic import plot_fitting_diagnostic

# ── 控制 from ivs_tools import * ──
__all__ = [
    "__version__",
    "HDF5Analyzer",
    "ConfigLoader",      
    "SchottkyAnalyzer",
    "SlidingWindowCheungFitter",
    "FittingConfig",
    "FittingResult"
]