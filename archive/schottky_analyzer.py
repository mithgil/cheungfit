import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ["Arial"]
rcParams['font.family'] = "sans-serif"

from scipy.signal import savgol_filter
from scipy import stats
import pandas as pd 
#from scipy.ndimage import median_filter # 另一種選擇

class SchottkyAnalyzer:
    def __init__(self, temperature=300.0, a_star=54.0):
        """
        temperature: 絕對溫度 (K)
        a_star: 理論 Richardson 常數 (A/cm^2·K^2)，MoS2 約為 54
        """
        self.T = temperature
        self.A_star = a_star
        self.k_B = 8.617333e-5  # Boltzmann constant in eV/K
        self.q = 1.6021766e-19  # Electron charge in C
        self.V_T = (1.380649e-23 * self.T) / self.q  # Thermal voltage (~0.0259V @ 300K) with k_B in unit of J/K

    def calculate_barrier(self, slope, intercept):
        """根據線性擬合結果計算物理參數"""
        # ln(J) = ln(J0) + (q / (n*k*T)) * V
        # slope = q / (n*k*T) = 1 / (n * V_T)
        ideality_factor = 1 / (slope * self.V_T)
        
        # J0 = A* * T^2 * exp(-phi_B / (k*T))
        # intercept = ln(J0)
        j0 = np.exp(intercept)
        phi_b = self.k_B * self.T * np.log((self.A_star * self.T**2) / j0)
        
        return ideality_factor, phi_b, j0

    def process_pixel(self, bias, current, area_cm2, fit_range=(0.1, 0.25), sg_window=13, median_window=7):
        """
        處理單點數據：Rolling Median -> S-G Filter -> Linear Fit
        """
        # 1. 單位轉換
        current_density = current / area_cm2
        
        # 2. 混合濾波處理
        # Step A: Rolling Median (去除尖銳雜訊/Outliers)
        # 使用 pandas 處理或是 scipy.ndimage.median_filter
        j_median = pd.Series(current_density).rolling(window=median_window, center=True).median().bfill().ffill().values
        
        # Step B: S-G 濾波 (平滑曲線)
        window = min(sg_window, len(j_median) // 2 * 2 - 1)
        
        if window < 5: # 確保數據長度足夠
            j_smooth = j_median
        else:
            j_smooth = savgol_filter(j_median, window, polyorder=3)
        
        # 3. 定義擬合遮罩 (確保 ln 前數據為正值)
        mask_fit = (bias >= fit_range[0]) & (bias <= fit_range[1] + 1e-9) & (j_smooth > 0)
    
        if np.sum(mask_fit) < 3:
            return None 
    
        v_fit = bias[mask_fit]
        ln_j_fit = np.log(j_smooth[mask_fit])
        
        # 4. 線性擬合
        slope, intercept, r_val, p_val, std_err = stats.linregress(v_fit, ln_j_fit)
        
        # 5. 計算物理參數 (n, phi_b, j0)
        n, phi_b, j0 = self.calculate_barrier(slope, intercept)
        
        return {
            'n': n, 'phi_b': phi_b, 'j0': j0, 
            'r2': r_val**2, 'slope': slope, 'intercept': intercept,
            'j_smooth': j_smooth, 'mask_fit': mask_fit
        }
        
    def process_pixel_adaptive(self, bias, current, area_cm2, window_size=15):
        current_density = current / area_cm2
        # 先做平滑處理
        j_median = pd.Series(current_density).rolling(window=5, center=True).median().fillna(method='bfill').values
        j_smooth = savgol_filter(j_median, 11, polyorder=3)
        
        ln_j = np.log(np.abs(j_smooth))
        
        best_res = None
        max_r2 = -1
    
        # 2. 定義滑動視窗搜尋
        # 設定搜尋範圍 0.05V 到 0.8V (避開 0V 附近的震盪)
        search_mask = (bias >= 0.05) & (bias <= 0.8)
        indices = np.where(search_mask)[0]
    
        for i in range(0, len(indices) - window_size, 3):
            idx_range = indices[i : i + window_size]
            v_sub = bias[idx_range]
            ln_j_sub = np.log(np.abs(j_smooth[idx_range]))
    
            slope, intercept, r_val, p_val, std_err = stats.linregress(v_sub, ln_j_sub)
            n, phi_b, j0 = self.calculate_barrier(slope, intercept)
    
            # 3. 關鍵篩選邏輯：
            # A. 避開 Plateau: n 必須在合理範圍 (例如 1.1 < n < 15)
            # B. 確保是正向增長: slope 必須 > 0
            # C. 在滿足 A & B 的視窗中，選 R2 最大的
            if 1.05 < n < 15.0 and slope > 0:
                if r_val**2 > max_r2:
                    max_r2 = r_val**2
                    best_res = {
                        'n': n, 'phi_b': phi_b, 'j0': j0, 'r2': r_val**2,
                        'slope': slope, 'intercept': intercept,
                        'v_range': (v_sub.min(), v_sub.max()),
                        'mask_fit': idx_range
                    }
        
        return best_res
    
def plot_schottky_analysis(bias, current, area_cm2, res, coords):
    x, y = coords
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: I-V Curve
    axes[0].plot(bias, current, 'b-', alpha=0.7, lw= 3, label='Raw Data')
    #axes[0].plot(bias, np.abs(current), 'b-', alpha=0.7, lw= 3, label='Raw Data')
    #axes[0].set_yscale('log') # 建議用對數坐標觀察
    axes[0].set_xlabel('Voltage (V)', fontsize = 15)
    axes[0].set_ylabel('Current (A)', fontsize = 15)
    axes[0].set_title(f'Point ({x}, {y}) Raw I-V', fontsize = 15)
    axes[0].tick_params(labelsize = 12)
    axes[0].grid(True, which="both", ls="-", alpha=0.4)

    # Right: Fitting ln(J) vs V
    j_density = current / area_cm2
    v_range = np.linspace(bias[res['mask_fit']].min(), bias.max(), 150)
    fit_line = res['intercept'] + res['slope'] * v_range
    
    axes[1].scatter(bias, np.log(np.abs(j_density)), s=60, c='gray', alpha=1, edgecolors='none', label='Raw ln(J)')
    axes[1].plot(bias, np.log(np.abs(res['j_smooth'])), 'ro', ms = 7, alpha=0.4, label='J smooth')
    axes[1].plot(v_range, fit_line, '--', c='navy', label=f'Linear Fit (R²={res["r2"]:.4f})')
    axes[1].tick_params(labelsize = 12)
    
    # 標示擬合區間
    axes[1].axvspan(bias[res['mask_fit']].min(), bias[res['mask_fit']].max(), color='green', alpha=0.1)
    
    # 參數文本
    textstr = '\n'.join((
        r'$\eta = %.2f$' % (res['n'], ),
        r'$\Phi_B = %.3f$ eV' % (res['phi_b'], ),
        r'$J_0$ = %.2e A/cm$^2$' % (res['j0'], )))
    
    axes[1].text(0.05, 0.95, textstr, transform=axes[1].transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    axes[1].set_xlabel('Voltage (V)', fontsize = 15)
    axes[1].set_ylabel('ln(J) (A/cm²)', fontsize = 15)
    axes[1].set_title('Schottky Analysis', fontsize = 15)
    axes[1].legend(fontsize = 12)
    axes[1].set_xlim(0.01, 1.1)
    autobalance_y(axes[1], (0.05, 1.1))
    plt.tight_layout()
    
    from pathlib import Path

    # 設定儲存路徑
    save_dir = Path("analysis_results/schottky_plots")
    save_dir.mkdir(parents=True, exist_ok=True) # 自動建立資料夾
    
    # 檔名包含座標，方便溯源
    file_name = f"point_{x}_{y}_fit.png"
    save_path = save_dir / file_name
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"圖表已儲存至: {save_path}")
    
    plt.show()

def autobalance_y(ax, x_range, padding=0.1):
    """
    根據指定的 x_range 自動調整 ax 的 ylim
    """
    lines = ax.get_lines()
    all_y = []
    for line in lines:
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        
        # 僅提取在 X 範圍內的 Y 值
        mask = (x_data >= x_range[0]) & (x_data <= x_range[1])
        pts = y_data[mask]
        # 過濾掉 inf 或 nan (取 log 容易出現)
        all_y.extend(pts[np.isfinite(pts)])
        
    if all_y:
        y_low, y_high = min(all_y), max(all_y)
        p = (y_high - y_low) * padding
        ax.set_ylim(y_low - p, y_high + p)

def plot_schottky_analysis_adaptive(bias, current, area_cm2, res, coords):
    x, y = coords
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # --- 左圖：原始 I-V 與半對數 I-V ---
    ax_left = axes[0]
    ax_left.plot(bias, np.abs(current), 'k-', alpha=0.3, label='Raw Abs(I)')
    # 使用雙 Y 軸或是直接在左圖改用 Log 座標觀察開啟行為
    ax_left.set_yscale('log') 
    ax_left.plot(bias, current, 'b-', lw=2, label='Current (A)')
    ax_left.set_xlabel('Voltage (V)', fontsize=14)
    ax_left.set_ylabel('Current (A) [Log Scale]', fontsize=14)
    ax_left.set_title(f'Point ({x}, {y}) conduction profile', fontsize=14)
    ax_left.grid(True, which="both", ls="-", alpha=0.2)
    ax_left.legend()

    # --- 右圖：ln(J) vs V 擬合分析 ---
    ax_right = axes[1]
    j_density = np.abs(current) / area_cm2
    ln_j_raw = np.log(j_density)
    ln_j_smooth = np.log(res['j_smooth'])

    # 畫出所有數據點
    ax_right.scatter(bias, ln_j_raw, s=30, c='gray', alpha=0.5, label='Raw ln(J)')
    ax_right.plot(bias, ln_j_smooth, 'r-', lw=1.5, alpha=0.8, label='Smoothed')

    # 取得擬合範圍的索引與電壓
    fit_mask = res['mask_fit']
    v_fit = bias[fit_mask]
    
    # 畫出擬合線 (延伸一點點方便觀察)
    v_ext = np.linspace(v_fit.min() - 0.05, v_fit.max() + 0.05, 100)
    fit_line = res['intercept'] + res['slope'] * v_ext
    ax_right.plot(v_ext, fit_line, 'g--', lw=3, label=f'Adaptive Fit (R²={res["r2"]:.4f})')

    # 亮綠色區塊：標示實際選中的擬合區間
    ax_right.axvspan(v_fit.min(), v_fit.max(), color='lime', alpha=0.2, label='Fit Window')
    
    # 參數文本
    textstr = '\n'.join((
        r'Selected Range: [%.2f, %.2f] V' % (v_fit.min(), v_fit.max()),
        r'$\eta = %.2f$' % (res['n'], ),
        r'$\Phi_B = %.3f$ eV' % (res['phi_b'], ),
        r'$J_0 = %.2e$ A/cm$^2$' % (res['j0'], )
    ))
    
    ax_right.text(0.05, 0.95, textstr, transform=ax_right.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    ax_right.set_xlabel('Voltage (V)', fontsize=14)
    ax_right.set_ylabel('ln(J) (A/cm²)', fontsize=14)
    ax_right.set_title(f'Schottky Fit (n={res["n"]:.1f})', fontsize=14)
    
    # 自動平衡 Y 軸：專注在有意義的數據範圍 (例如過濾掉極低電流的雜訊區)
    autobalance_y(ax_right, (bias.min(), bias.max()))
    
    ax_right.legend(loc='lower right', fontsize=10)
    ax_right.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # 儲存與顯示 (保持你原本的邏輯)
    from pathlib import Path
    save_dir = Path("analysis_results/adaptive_plots")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"point_{x}_{y}_adaptive.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
# --- 執行部分 ---
if __name__ == "__main__":
    
    analyzer = SchottkyAnalyzer(temperature=300)
    d_nm = 25
    area_cm2 = np.pi * (d_nm/2 * 1e-7)**2 # 修正：1nm = 1e-7 cm
    
    x, y = 62, 68
    spectrum = h5map.get_spectrum_at_pixel(x, y)[3:] # 跳過前3點
    bias = h5map.bias_vector[3:]
    
    results = analyzer.process_pixel(bias, spectrum, area_cm2)
    if results:
        plot_schottky_analysis(bias, spectrum, area_cm2, results, (x, y))
        