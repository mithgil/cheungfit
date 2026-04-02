# 清除快取
find src/ -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find src/ -name "*.pyc" -delete 2>/dev/null

# 重新安裝
pip install -e . --force-reinstall --no-deps

# 測試
python -c "
from ivs_tools.physics.sliding_window_fitter import FittingConfig
from ivs_tools.config_loader import ConfigLoader

# 測試 YAML 載入
cfg_dict = ConfigLoader.get_fitting_config('monolayer')
print('YAML 載入的類型:')
for k, v in cfg_dict.items():
    print(f'   {k}: {type(v).__name__} = {v}')

# 測試 FittingConfig
cfg = FittingConfig(**cfg_dict)
print(f'\nFittingConfig 初始化後:')
print(f'   n_min: {type(cfg.n_min).__name__} = {cfg.n_min}')
print(f'   n_max: {type(cfg.n_max).__name__} = {cfg.n_max}')
print(f'   window_size: {type(cfg.window_size).__name__} = {cfg.window_size}')
print('✅ 類型轉換成功!')
"