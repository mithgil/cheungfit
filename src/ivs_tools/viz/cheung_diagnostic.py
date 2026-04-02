# src/ivs_tools/viz/cheung_diagnostic.py
"""
Diagnostic visualization for Cheung method fitting
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from typing import Optional, Tuple
from pathlib import Path
from IPython.display import display, Markdown

def plot_fitting_diagnostic(fitter, result,
                           pixel_coords: Tuple[int, int],
                           save_path: Optional[str] = None,
                           xlim_bias: Tuple[float, float] = (0, 1),
                           ylim_n: Tuple[float, float] = (2, 8),
                           show_summary: bool = True):
    """
    繪製完整擬合診斷圖 (5 panels)
    
    Top row (1,3): I-V semilogy, ln(I)-V, n histogram
    Bottom row (1,2): n vs bias window, R² vs bias window
    
    Parameters
    ----------
    fitter : SlidingWindowCheungFitter
        已執行 fit() 的擬合器
    result : FittingResult
        擬合結果物件
    pixel_coords : tuple
        (x, y) 像素座標
    save_path : str, optional
        儲存路徑
    xlim_bias : tuple
        X 軸偏壓範圍 (min, max)
    show_summary : bool
        是否顯示 Markdown 結果摘要
    """
    # 設定 matplotlib 樣式
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 12,
        'axes.labelsize': 15,
        'axes.titlesize': 15,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 10,
        'figure.titlesize': 20,
        'figure.titleweight': 'bold'
    })
    
    diag = fitter.get_diagnostic_data()
    if not diag:
        print("⚠️ 無診斷數據，請先執行 fit()")
        return
    
    # ========== 顯示 Markdown 結果摘要 ==========
    if show_summary:
        _display_summary_markdown(result, diag, pixel_coords)
    
    # ========== 建立畫布：2 rows x 2 columns ==========
    fig = plt.figure(figsize=(16, 10))
    # GridSpec: 2 rows, 2 columns (等寬)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    bias = diag.get('bias', [])
    I = diag.get('I', [])
    n_samples = diag.get('n_samples', [])
    r2_samples = diag.get('r2_samples', [])
    n_vs_V = diag.get('n_vs_V', [])
    r2_vs_V = diag.get('r2_vs_V', [])
    
    # ========== Top Row (3 panels in first row) ==========
    # 使用嵌套 GridSpec 在第一行放 3 個 panel
    gs_top = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, :])
    
    # Panel 1: I-V (semilogy)
    ax1 = fig.add_subplot(gs_top[0])
    if len(bias) > 0 and len(I) > 0:
        ax1.semilogy(bias, I, 'ko', ms=4, alpha=0.3, label='Full I-V')
        if result.fit_indices:
            idx_start, idx_end = result.fit_indices
            ax1.semilogy(bias[idx_start:idx_end], I[idx_start:idx_end], 
                        'ro', ms=7, alpha=0.3, label='Fitting window')
    ax1.set_xlabel('Bias (V)')
    ax1.set_ylabel('|I| (A)')
    ax1.set_title('Panel 1: I-V Curve')
    ax1.set_xlim(xlim_bias)
    ax1.legend(loc='lower right')
    ax1.grid(True, ls='--', alpha=0.3)
    
    # Panel 2: ln(I) vs V
    ax2 = fig.add_subplot(gs_top[1])
    if len(bias) > 0 and len(I) > 0:
        ln_I = np.log(I + 1e-20)
        ax2.plot(bias, ln_I, 'ko', ms=4, alpha=0.3, label='Full')
        if result.fit_indices:
            idx_start, idx_end = result.fit_indices
            ax2.plot(bias[idx_start:idx_end], ln_I[idx_start:idx_end], 
                    'ro', ms=7, alpha=0.3, label='Fit region')
    ax2.set_xlabel('Bias (V)')
    ax2.set_ylabel('ln(|I|)')
    ax2.set_title('Panel 2: ln(I) vs V')
    ax2.set_xlim(xlim_bias)
    ax2.legend(loc='lower right')
    ax2.grid(True, ls='--', alpha=0.3)
    
    # Panel 3: n histogram
    ax3 = fig.add_subplot(gs_top[2])
    if n_samples:
        ax3.hist(n_samples, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
        if result.n:
            ax3.axvline(x=result.n, color='red', ls='--', lw=2, label=f'n={result.n:.2f}')
        ax3.axvline(x=np.median(n_samples), color='green', ls=':', lw=2, 
                   label=f'median={np.median(n_samples):.2f}')
        ax3.axvline(x=fitter.config.n_min, color='orange', ls='-.', lw=1, label='n_min')
        ax3.axvline(x=fitter.config.n_max, color='orange', ls='-.', lw=1, label='n_max')
    ax3.set_xlabel('Extracted n')
    ax3.set_ylabel('Count')
    ax3.set_title('Panel 3: n Distribution')
    ax3.legend()
    ax3.grid(True, ls='--', alpha=0.3)
    
    # ========== Bottom Row (2 panels, equal width) ==========
    
    # Panel 4: n vs bias window (left half)
    ax4 = fig.add_subplot(gs[1, 0])
    if n_vs_V:
        V_centers_n, n_vals = zip(*n_vs_V)
        ax4.scatter(V_centers_n, n_vals, s=50, alpha=0.9, c='tab:blue', 
                   edgecolors='black', linewidth=0.5, zorder=2, label='All windows')
        if result.fit_range:
            best_V_center = sum(result.fit_range) / 2
            ax4.scatter([best_V_center], [result.n], 
                       s=150, c='red', marker='*', edgecolors='black', 
                       linewidth=1.5, zorder=3, label='Best')
            ax4.axhline(y=result.n, color='red', ls='--', alpha=0.7, zorder=1)
    ax4.set_xlabel('Bias Window Center (V)')
    ax4.set_ylabel('Extracted n')
    ax4.set_ylim(ylim_n)
    ax4.set_xlim(xlim_bias)
    ax4.set_title('Panel 4: n vs Bias Window')
    ax4.legend(loc='upper right')
    ax4.grid(True, ls='--', alpha=0.3, zorder=0)
    
    # Panel 5: R² vs bias window (right half)
    ax5 = fig.add_subplot(gs[1, 1])
    if r2_vs_V:
        V_centers_r2, r2_vals = zip(*r2_vs_V)
        ax5.scatter(V_centers_r2, r2_vals, s=50, alpha=0.9, c='tab:orange', 
                   edgecolors='black', linewidth=0.5, zorder=2, label='All windows')
        if result.fit_range:
            best_V_center = sum(result.fit_range) / 2
            ax5.scatter([best_V_center], [result.r_squared], 
                       s=150, c='red', marker='*', edgecolors='black', 
                       linewidth=1.5, zorder=3, label=f'Best: {result.r_squared:.4f}')
            ax5.axhline(y=result.r_squared, color='red', ls='--', alpha=0.7, zorder=1)
            ax5.axvline(x=best_V_center, color='red', ls=':', alpha=0.3, zorder=1)
        ax5.axhline(y=fitter.config.min_r_squared, color='gray', ls='-.', 
                   alpha=0.5, label=f'Threshold: {fitter.config.min_r_squared:.2f}')
    ax5.set_xlabel('Bias Window Center (V)')
    ax5.set_ylabel('R²')
    ax5.set_ylim(0.7, 1.1)
    ax5.set_xlim(xlim_bias)
    ax5.set_title('Panel 5: R² vs Bias Window')
    ax5.legend(loc='lower right')
    ax5.grid(True, ls='--', alpha=0.3, zorder=0)
    
    # 總標題
    plt.suptitle(f'Cheung Fitting Diagnostic - Pixel {pixel_coords}', 
                fontsize=20, fontweight='bold', y=1.02)
    
    if save_path:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Diagnostic plot saved: {save_path}")
    
    return fig


def _display_summary_markdown(result, diag: dict, pixel_coords: Tuple[int, int]):
    """
    以 Markdown 格式顯示結果摘要（Jupyter Lab 渲染）
    """
    stats = diag.get('stats_counter', {})
    
    if result.success:
        markdown_text = f"""
### ✅ FITTING SUCCESS - Pixel {pixel_coords}

| Parameter | Value |
|-----------|-------|
| **ΦB (SBH)** | {result.phi_b:.3f} eV |
| **n (ideality)** | {result.n:.2f} |
| **Rs (series resistance)** | {result.Rs:.2e} Ω |
| **R²** | {result.r_squared:.4f} |
| **Score** | {result.score:.3f} |

**Fit Range:** {result.fit_range[0]:.2f} ~ {result.fit_range[1]:.2f} V

**Statistics:**
- Total windows scanned: {stats.get('total', 'N/A')}
- Monotonic fail: {stats.get('monotonic_fail', 'N/A')}
- R² pre-filter fail: {stats.get('r2_pre_fail', 'N/A')}
- Constraint fail: {stats.get('constraint_fail', 'N/A')}
- R² final fail: {stats.get('r2_final_fail', 'N/A')}
- Rs mismatch: {stats.get('rs_mismatch', 'N/A')}
- ✅ **Passed all checks: {stats.get('success', 'N/A')}**
"""
    else:
        markdown_text = f"""
### ❌ FITTING FAILED - Pixel {pixel_coords}

**Reason:** {result.warning if result.warning else 'Unknown'}

**Suggestions:**
- Check data quality (noise, spikes)
- Adjust current_threshold
- Try different layer_type setting
- Consider manual range selection
"""
    
    display(Markdown(markdown_text))