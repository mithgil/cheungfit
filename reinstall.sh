find src/ -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
find src/ -name "*.pyc" -delete 2>/dev/null

# 重新安裝
pip install -e . --force-reinstall --no-deps