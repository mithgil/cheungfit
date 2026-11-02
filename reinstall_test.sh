#!/usr/bin/env bash
# 重新安裝 ivs_tools 並執行基本 smoke test。
# 預設使用 sci 環境的 Python，避免裸 pip 安裝到錯誤的環境。
set -euo pipefail

cd "$(dirname "$0")"

PYTHON="${PYTHON:-$HOME/micromamba/envs/sci/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
    echo "錯誤：找不到 Python 執行檔：$PYTHON" >&2
    echo "請設定 PYTHON=/path/to/python 後重試。" >&2
    exit 1
fi

echo "使用 Python: $("$PYTHON" -c 'import sys; print(sys.executable)')"

# 清除快取
find src/ -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
find src/ -type f -name "*.pyc" -delete 2>/dev/null || true

# 重新安裝
"$PYTHON" -m pip install -e . --force-reinstall --no-deps

# 測試
"$PYTHON" - <<'PY'
from ivs_tools.physics.sliding_window_fitter import FittingConfig
from ivs_tools.config_loader import ConfigLoader

# 測試 YAML 載入
cfg_dict = ConfigLoader.get_fitting_config('monolayer')
print('YAML 載入的類型:')
for k, v in cfg_dict.items():
    print(f'   {k}: {type(v).__name__} = {v}')

# 測試 FittingConfig
cfg = FittingConfig(**cfg_dict)
print('\nFittingConfig 初始化後:')
print(f'   n_min: {type(cfg.n_min).__name__} = {cfg.n_min}')
print(f'   n_max: {type(cfg.n_max).__name__} = {cfg.n_max}')
print(f'   window_size: {type(cfg.window_size).__name__} = {cfg.window_size}')
print('✅ 類型轉換成功!')
PY
