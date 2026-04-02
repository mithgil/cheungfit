from ivs_tools import HDF5Analyzer, SchottkyAnalyzer, SlidingWindowCheungFitter
from ivs_tools import FittingConfig, FittingResult
print('✅ All imports OK!')
print(f'   HDF5Analyzer: {HDF5Analyzer.__module__}')
print(f'   SchottkyAnalyzer: {SchottkyAnalyzer.__module__}')
print(f'   SlidingWindowCheungFitter: {SlidingWindowCheungFitter.__module__}')
"""
Expected:

✅ All imports OK!
   HDF5Analyzer: ivs_tools.io.h5utils
   SchottkyAnalyzer: ivs_tools.physics.schottky_analyzer
   SlidingWindowCheungFitter: ivs_tools.physics.sliding_window_fitter
"""

from ivs_tools import ConfigLoader
area = ConfigLoader.get_area('cafm_tip_10nm')
print(f'✅ Area loaded: {area:.3e} cm²')

#✅ Area loaded: 7.854e-13 cm²

from ivs_tools import ConfigLoader, SchottkyAnalyzer
area = ConfigLoader.get_area('cafm_tip_10nm')
analyzer = SchottkyAnalyzer(area_cm2=area)
print('✅ Analyzer initialized with YAML config!')

#✅ Analyzer initialized with YAML config!