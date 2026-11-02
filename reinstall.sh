#!/usr/bin/env bash
# 重新以 editable 模式安裝 ivs_tools。
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
