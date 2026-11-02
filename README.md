# cheungfit / ivs_tools

Nanonis SPM I–V 數據處理與 Schottky/Cheung 分析工具。

`ivs_tools` 提供：

- `HDF5Analyzer`：讀取 Nanonis `.h5` 數據
- `SchottkyAnalyzer`：Schottky 能障分析
- `ConfigLoader`：載入 `config/*.yml` 分析設定
- `SlidingWindowCheungFitter`：Cheung 擬合與 sliding-window 分析
- `ivs_tools.viz.cheung_diagnostic`：擬合診斷圖

## 安裝

本專案使用 `src/` layout，必須以 editable 模式安裝後才能 `import ivs_tools`。

```bash
# 確認使用 sci 環境的 Python（不要用裸 pip）
/home/tim/micromamba/envs/sci/bin/python -m pip install -e .

# 或直接使用重裝腳本
./reinstall.sh
```

`reinstall.sh`、`reinstall_test.sh`、`upgrade.sh` 預設使用
`$HOME/micromamba/envs/sci/bin/python`；若你的環境路徑不同，可用 `PYTHON` 覆寫：

```bash
PYTHON=/path/to/python ./reinstall.sh
```

## 使用

```python
from ivs_tools import HDF5Analyzer, SchottkyAnalyzer, ConfigLoader
from ivs_tools.viz.cheung_diagnostic import plot_fitting_diagnostic
```

在 Jupyter 中記得選擇 **Python (sci)** kernel，安裝或更新套件後需重啟 kernel。

## 專案結構

```text
src/ivs_tools/     # 套件原始碼
config/            # YAML 分析設定
data/              # 原始與處理後數據
notebooks/         # 分析 notebook
tests/             # 測試
```

## 開發

```bash
/home/tim/micromamba/envs/sci/bin/python -m pip install -e ".[dev]"
/home/tim/micromamba/envs/sci/bin/python -m pytest
```
