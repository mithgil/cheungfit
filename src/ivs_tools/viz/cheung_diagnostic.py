# src/ivs_tools/viz/cheung_diagnostic.py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def plot_fitting_diagnostic(fitter: SlidingWindowCheungFitter, 
                           pixel_coords: Tuple[int,int],
                           save_path: Optional[str] = None):
    """
    繪製擬合診斷圖：幫助判斷是否需人工介入
    """
    diag = fitter.get_diagnostic_plot_data()
    if not diag:
        print("⚠️ 無診斷數據，請先執行 fit()")
        return
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig)
    
    # Panel 1: 原始 I-V + 擬合區間標記
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(diag['bias'], diag['I'], 'k-', lw=0.5, label='Full I-V')
    if fitter.last_diagnostic and 'window_I' in diag:
        ax1.semilogy(diag['bias'][fitter.last_diagnostic['fit_indices'][0]:
                                   fitter.last_diagnostic['fit_indices'][1]], 
                    diag['window_I'], 'r-', lw=2, label='Fitting window')
    ax1.set_xlabel('Bias (V)')
    ax1.set_ylabel('Current (A)')
    ax1.set_title(f'Pixel {pixel_coords}: I-V Curve')
    ax1.legend(fontsize=8)
    ax1.grid(True, ls='--', alpha=0.3)
    
    # Panel 2: Cheung Function I + 擬合線
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.scatter(diag['I'], diag['dV_dlnI'], s=3, c='gray', alpha=0.5, label='All points')
    if 'window_I' in diag:
        # 擬合線
        I_fit = np.linspace(diag['window_I'].min(), diag['window_I'].max(), 50)
        y_fit = fitter.last_diagnostic.get('slope_a', 0) * I_fit + fitter.last_diagnostic.get('intercept_a', 0)
        ax2.plot(I_fit, y_fit, 'r--', lw=2, label='Linear fit')
        ax2.scatter(diag['window_I'], diag['window_y'], s=10, c='red', label='Window')
    ax2.set_xlabel('Current (A)')
    ax2.set_ylabel('dV/d(ln I)')
    ax2.set_title('Cheung Function I')
    ax2.legend(fontsize=8)
    ax2.grid(True, ls='--', alpha=0.3)
    
    # Panel 3: Cheung Function II (H-function)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(diag['I'], diag.get('H_I_full', diag['bias']), s=3, c='gray', alpha=0.5)
    if 'window_I' in diag:
        H_fit = fitter.last_diagnostic.get('slope_b', 0) * diag['window_I'] + fitter.last_diagnostic.get('intercept_b', 0)
        ax3.plot(diag['window_I'], H_fit, 'b--', lw=2)
        ax3.scatter(diag['window_I'], diag['window_H'], s=10, c='blue')
    ax3.set_xlabel('Current (A)')
    ax3.set_ylabel('H(I) (V)')
    ax3.set_title('Cheung Function II')
    ax3.grid(True, ls='--', alpha=0.3)
    
    # Panel 4: 被過濾的候選區間統計（幫助理解為何某些區間被排除）
    ax4 = fig.add_subplot(gs[1, :])
    candidates = diag.get('candidates', [])
    if candidates:
        n_vals = [c['n'] for c in candidates]
        r2_vals = [c['r2'] for c in candidates]
        colors = ['red' if 'n' in c.get('reason','') else 'orange' for c in candidates]
        ax4.scatter(n_vals, r2_vals, c=colors, alpha=0.6, s=20)
        ax4.axvline(x=fitter.config.n_min, color='green', ls=':', label='n bounds')
        ax4.axvline(x=fitter.config.n_max, color='green', ls=':')
        ax4.axhline(y=fitter.config.min_r_squared, color='green', ls=':', label='min R²')
        ax4.set_xlabel('Extracted n')
        ax4.set_ylabel('R² of window')
        ax4.set_title('Rejected Windows Diagnostic')
        ax4.legend(fontsize=8)
        ax4.grid(True, ls='--', alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No rejected windows (all passed or none evaluated)', 
                ha='center', va='center', transform=ax4.transAxes)
    
    # Panel 5: 結果摘要
    ax5 = fig.add_subplot(gs[1, -1])
    ax5.axis('off')
    if fitter.last_diagnostic and fitter.last_diagnostic.get('success'):
        res = fitter.last_diagnostic
        summary = f"""
        ✅ FITTING SUCCESS
        
        Pixel: {pixel_coords}
        ─────────────────
        η (n)   = {res.get('n', 'N/A'):.3f}
        Rs      = {res.get('Rs', 'N/A'):.2e} Ω
        ΦB      = {res.get('phi_b', 'N/A'):.3f} eV
        R²      = {res.get('r_squared', 'N/A'):.4f}
        Score   = {res.get('score', 'N/A'):.3f}
        
        Fit Range:
        V: {res.get('fit_range', (None,None))[0]:.2f} ~ {res.get('fit_range', (None,None))[1]:.2f} V
        
        Config:
        Window: {fitter.config.window_size} pts
        Layer: {fitter.config.layer_type}
        """
    else:
        summary = f"""
        ❌ FITTING FAILED
        
        Reason: {fitter.last_diagnostic.get('warning', 'Unknown') if fitter.last_diagnostic else 'No result'}
        
        Suggestions:
        • Check data quality (noise, spikes)
        • Adjust current_threshold
        • Try different layer_type setting
        • Consider manual range selection
        """
    ax5.text(0.05, 0.95, summary, fontsize=9, family='monospace',
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.9))
    
    plt.suptitle(f'Cheung Fitting Diagnostic - Pixel {pixel_coords}', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Diagnostic plot saved: {save_path}")
    
    return fig