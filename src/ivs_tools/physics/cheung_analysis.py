

import matplotlib.pyplot as plt
import numpy as np

from numba import njit

@njit(cache=True)
def _numba_sliding_window_core(current_I, current_Ya, V_T, window_size):

    n_pts = len(current_I)
    
    # 存儲多個候選結果
    candidates = []
    
    for i in range(n_pts - window_size):
        x = current_I[i : i + window_size]
        y = current_Ya[i : i + window_size]
        
        # 線性擬合
        n = window_size
        sum_x = np.sum(x)
        sum_y = np.sum(y)
        sum_xx = np.dot(x, x)
        sum_xy = np.dot(x, y)
        sum_yy = np.dot(y, y)
        
        denom = (n * sum_xx - sum_x**2)
        if denom == 0: 
            continue
            
        m = (n * sum_xy - sum_x * sum_y) / denom
        c = (sum_y - m * sum_x) / n
        
        # 計算 R²
        num_r = (n * sum_xy - sum_x * sum_y)
        den_r2 = (n * sum_xx - sum_x**2) * (n * sum_yy - sum_y**2)
        r2 = (num_r**2) / den_r2 if den_r2 > 0 else 0
        
        # 計算標準誤差
        y_pred = m * x + c
        residuals = y - y_pred
        mse = np.sum(residuals**2) / (n - 2) if n > 2 else 0
        
        # 計算 ideality factor
        n_tmp = c / V_T
        
        candidates.append({
            'r2': r2,
            'mse': mse,
            'slope': m,
            'intercept': c,
            'n_tmp': n_tmp,
            'start_idx': i
        })
    
    # 如果沒有候選結果
    if len(candidates) == 0:
        return 0.0, 0.0, -1.0, -1
    
    # 按多個標準排序（加權評分）
    best_score = -1.0
    best_idx = -1
    
    for j, cand in enumerate(candidates):
        # 綜合評分：R² 為主，MSE 為輔
        score = cand['r2'] * 0.8 + (1.0 / (1.0 + cand['mse'])) * 0.2 if cand['mse'] > 0 else cand['r2']
        
        # 物理約束：斜率正、ideality factor 合理
        if cand['slope'] > 0 and 1.0 < cand['n_tmp'] < 10.0:
            if score > best_score:
                best_score = score
                best_idx = j
    
    if best_idx == -1:
        return 0.0, 0.0, -1.0, -1
    
    best = candidates[best_idx]
    return best['slope'], best['intercept'], best['r2'], best['start_idx']

def advanced_cheung_analysis(bias, current, area_cm2, T=300, a_star=54.0, 
                           window_fraction=0.1, min_r2=0.95):
    """
    嘗試多個窗口大小尋找最佳擬合
    """
    # 基本物理常數
    V_T = 8.617333e-5 * T
    constant_term = area_cm2 * a_star * (T**2)
    
    # 1. 輸入數據預處理和驗證
    bias = np.asarray(bias, dtype=np.float64)
    current = np.asarray(current, dtype=np.float64)
    
    # 檢查數據長度
    assert len(bias) == len(current), "bias and current must have same length"
    
    # 2. 數據平滑處理
    abs_I = np.abs(current)
    
    # 使用滾動中值去除尖峰
    pad_size = 7 // 2
    padded_I = np.pad(abs_I, pad_size, mode='edge')
    
    # 使用滑動窗口視圖計算中值
    sliding_view = np.lib.stride_tricks.sliding_window_view(padded_I, 7)
    I_median = np.median(sliding_view, axis=-1)
    
    # Savitzky-Golay 濾波
    window_length = min(31, len(I_median))
    if window_length % 2 == 0:  # 確保奇數
        window_length -= 1
    if window_length >= 5:  # 最小窗口大小
        I_smooth = savgol_filter(I_median, window_length=window_length, polyorder=3)
    else:
        I_smooth = I_median.copy()
    
    # 3. 計算 dV/d(lnI)
    ln_I = np.log(np.maximum(I_smooth, 1e-20))  # 避免 log(0)
    y_func_a = np.gradient(bias) / np.gradient(ln_I)
    
    # 4. 確定搜索區域
    noise_floor = np.max(I_smooth) * 0.01
    search_mask = (bias > 0.2) & (I_smooth > noise_floor)
    indices = np.where(search_mask)[0]
    
    if len(indices) < 10:
        return create_empty_res(bias, y_func_a, error="Not enough data points in search region")
    
    # 5. 嘗試不同的窗口大小
    base_window = max(10, int(len(indices) * window_fraction))
    
    # 創建窗口大小候選列表
    window_candidates = []
    for factor in [0.5, 1.0, 2.0]:
        ws = int(base_window * factor)
        if 10 <= ws <= len(indices):
            window_candidates.append(ws)
    
    # 去重並排序
    window_candidates = sorted(set(window_candidates))
    
    best_results = {
        'r2': -1.0,
        'window_size': 0,
        'm_a': 0.0,
        'c_a': 0.0,
        'start_idx': -1
    }
    
    # 6. 對每個窗口大小進行搜索
    for ws in window_candidates:
        m_a, c_a, r2, start_off = _numba_sliding_window_core(
            I_smooth[indices], y_func_a[indices], V_T, ws
        )
        
        # 確保返回值是有效的
        if start_off == -1:
            continue
            
        # 類型轉換確保是整數
        start_off = int(start_off)
        
        # 計算 ideality factor
        n_tmp = c_a / V_T
        
        # 檢查結果質量
        is_valid = (
            r2 > min_r2 and 
            m_a > 0 and 
            1.0 < n_tmp < 10.0 and  # 合理的理想因子範圍
            start_off >= 0 and
            start_off + ws <= len(indices)
        )
        
        # 如果找到更好的結果，更新
        if is_valid and r2 > best_results['r2']:
            best_results.update({
                'r2': float(r2),
                'window_size': int(ws),
                'm_a': float(m_a),
                'c_a': float(c_a),
                'start_idx': int(start_off)  # 確保是整數
            })
    
    # 7. 檢查是否找到有效結果
    if best_results['r2'] < min_r2 or best_results['start_idx'] == -1:
        error_msg = f"No valid fit found (best R²={best_results['r2']:.3f})"
        return create_empty_res(bias, y_func_a, error=error_msg)
    
    # 8. 計算結果參數
    n_a = best_results['c_a'] / V_T
    Rs_a = best_results['m_a']
    ws = int(best_results['window_size'])  # 再次確保是整數
    start_off = int(best_results['start_idx'])  # 再次確保是整數
    
    # 檢查索引範圍是否有效
    if start_off < 0 or start_off + ws > len(indices):
        return create_empty_res(bias, y_func_a, error="Invalid index range")
    
    # 找到對應的索引範圍
    idx_fit = indices[start_off : start_off + ws]
    
    # 確保索引是整數陣列
    idx_fit = np.array(idx_fit, dtype=int)
    
    # 9. 計算 H(I) 和方法 B 參數
    # 數值穩定性處理
    log_arg = I_smooth / constant_term
    log_arg = np.maximum(log_arg, 1e-15)  # 避免 log(0) 或負值
    H_I = bias - (n_a * V_T * np.log(log_arg))
    
    # 方法 B 的線性擬合
    slope_b, intercept_b, r_val_b, _, _ = stats.linregress(
        I_smooth[idx_fit], H_I[idx_fit]
    )
    
    phi_b = intercept_b / n_a
    
    # 10. 返回完整結果
    return {
        'n': float(n_a),
        'Rs_a': float(Rs_a),
        'phi_b': float(phi_b),
        'Rs_b': float(slope_b),
        'r2_a': float(best_results['r2']),
        'r2_b': float(r_val_b**2),
        'fit_range': (float(bias[idx_fit[0]]), float(bias[idx_fit[-1]])),
        'indices': idx_fit,
        'window_size': ws,
        'H_I': H_I,
        'y_a': y_func_a,
        'I_smooth': I_smooth,
        'bias': bias,  # 保存原始數據
        'current': current,
        'success': True,
        'method': f"Multi-window search (ws={ws})",
        'fit_region_size': ws,
        'fit_quality': 'good' if best_results['r2'] > 0.98 else 'acceptable'
    }


def create_empty_res(bias, y_func_a, error="Unknown error"):
    """創建空的結果字典"""
    return {
        'n': np.nan, 
        'Rs_a': np.nan, 
        'phi_b': np.nan, 
        'Rs_b': np.nan,
        'r2_a': 0.0, 
        'r2_b': 0.0,
        'fit_range': (np.nan, np.nan),
        'indices': np.array([], dtype=int),
        'window_size': 0,
        'H_I': np.full_like(bias, np.nan),
        'y_a': y_func_a,
        'I_smooth': np.full_like(bias, np.nan),
        'bias': bias,
        'current': np.full_like(bias, np.nan),
        'success': False,
        'method': 'No fit found',
        'fit_region_size': 0,
        'fit_quality': 'poor',
        'error': error
    }

def plot_cheung_results(bias, current, result, figsize=(12, 8)):
    """
    繪製 Cheung 分析結果的完整圖表
    - Method A: subplot(2,3,4)
    - Method B: subplot(2,3,5)
    """
    if not result['success']:
        print("無法繪製：分析失敗")
        return None
    
    fig = plt.figure(figsize=figsize)
    
    # 1. 原始 I-V 曲線（線性座標）
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(bias, current * 1e6, 'bo', markersize = 3, alpha=0.6, label='Original')
    ax1.plot(bias, result['I_smooth'] * 1e6, 'ro', markersize = 3, alpha=0.6, label='Smoothed')
    
    # 標記擬合區域
    if len(result['indices']) > 0:
        fit_bias = bias[result['indices']]
        fit_current = current[result['indices']]
        
        # 添加擬合區域陰影
        fit_min, fit_max = result['fit_range']
        ax1.axvspan(fit_min, fit_max, alpha=0.2, color='green', label=f'Fit range: {fit_min:.3f}-{fit_max:.3f} V')
    
    ax1.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax1.set_xlabel('Bias Voltage (V)', fontsize=12)
    ax1.set_ylabel('Current (µA)', fontsize=12)
    ax1.set_title('I-V Characteristic', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 2. 原始 I-V 曲線（半對數座標）
    ax2 = plt.subplot(2, 2, 2)
    pos_mask = bias > 0
    neg_mask = bias < 0
    
    if np.any(pos_mask):
        ax2.semilogy(bias[pos_mask], np.abs(current[pos_mask]) * 1e6, 'b-', 
                    linewidth=1.5, label='Forward')
    if np.any(neg_mask):
        ax2.semilogy(-bias[neg_mask], np.abs(current[neg_mask]) * 1e6, 'r-', 
                    linewidth=1.5, label='Reverse')
    
    # 添加擬合區域陰影
        fit_min, fit_max = result['fit_range']
        ax2.axvspan(fit_min, fit_max, alpha=0.2, color='green', label=f'Fit range: {fit_min:.3f}-{fit_max:.3f} V')
    
    ax2.set_xlabel('|Bias Voltage| (V)', fontsize=12)
    ax2.set_ylabel('|Current| (µA)', fontsize=12)
    ax2.set_title('Semi-log I-V Plot', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, which='both')
    
    
    # 4. 方法 A: dV/d(lnI) vs I (位置 4)
    ax3 = plt.subplot(2, 2, 3)
    
    # 獲取擬合區域的索引
    if len(result['indices']) > 0:
        idx_fit = result['indices']
        
        # 計算擬合區域前後 x% 的數據點
        n_total = len(bias)
        margin = int(0.013 * n_total)  # 2.% 的數據點
        
        # 創建擴展的顯示區域
        start_extended = max(0, idx_fit[0] - margin)
        end_extended = min(n_total, idx_fit[-1] + margin)
        idx_extended = np.arange(start_extended, end_extended)
        
        # 繪製擴展區域的數據（灰色）
        ax3.plot(bias[idx_extended], result['y_a'][idx_extended], 
                'o', color='gray', linewidth=1, alpha=0.5, label='Data')
        
        # 繪製擬合區域的數據（藍色）
        ax3.plot(bias[idx_fit], result['y_a'][idx_fit], 'bo', 
                markersize=6, label='Fit points')
        
        # 繪製擬合線（紅色）
        I_fit = result['I_smooth'][idx_fit]
        y_fit = result['Rs_a'] * I_fit + result['n'] * 8.617e-5 * 300  # n*V_T
        
        ax3.plot(bias[idx_fit], y_fit, 'r-', linewidth=2, 
                label=f'Linear fit (R²={result["r2_a"]:.4f})')
        
        # 添加擬合方程式
        fit_eq = f'y = {result["Rs_a"]:.1f}·I + {result["n"]*8.617e-5*300:.3f}'
        ax3.text(0.05, 0.95, fit_eq, transform=ax3.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax3.set_xlabel('Bias Voltage (V)', fontsize=12)
    ax3.set_ylabel('dV/d(lnI) (V)', fontsize=12)
    ax3.set_title(f'Cheung Method A\nn = {result["n"]:.3f}, Rs = {result["Rs_a"]:.1f} Ω', 
                 fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 5. 方法 B: H(I) vs I 
    ax4 = plt.subplot(2, 2, 4)
    
    if len(result['indices']) > 0:
        idx_fit = result['indices']
        I_abs = np.abs(result['I_smooth'])
        H_I = result['H_I']
        
        # 計算擴展顯示區域
        start_extended = max(0, idx_fit[0] - margin)
        end_extended = min(n_total, idx_fit[-1] + margin)
        idx_extended = np.arange(start_extended, end_extended)
        
        # 繪製擴展區域的數據（灰色）
        ax4.plot(I_abs[idx_extended] * 1e6, H_I[idx_extended], 
                'o', color='gray', linewidth=1, alpha=0.5, label='Data')
        
        # 繪製擬合區域的數據（藍色）
        ax4.plot(I_abs[idx_fit] * 1e6, H_I[idx_fit], 'bo', 
                markersize=6, label='Fit points')
        
        # 繪製擬合線（紅色）
        I_fit_abs = np.abs(I_fit)
        H_fit = result['Rs_b'] * I_fit + result['n'] * result['phi_b']
        
        ax4.plot(I_fit_abs * 1e6, H_fit, 'r-', linewidth=2,
                label=f'Linear fit (R²={result["r2_b"]:.4f})')
        
        # 添加擬合方程式
        fit_eq = f'H = {result["Rs_b"]:.1f}·I + {result["n"]*result["phi_b"]:.3f}'
        ax4.text(0.05, 0.95, fit_eq, transform=ax4.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax4.set_xlabel('Current (µA)', fontsize=12)
    ax4.set_ylabel('H(I) (V)', fontsize=12)
    ax4.set_title(f'Cheung Method B\n' + fr'$\Phi_b$ = {result["phi_b"]:.3f} eV, $R_s$ = {result["Rs_b"]:.1f} $\Omega$', 
             fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 添加整體標題
    fig.suptitle(f'Schottky Barrier Analysis - Pixel ({xs}, {ys})', 
                fontsize=16, fontweight='bold', y=1.02)
    
    return fig

def print_simple_table(result):
    """打印簡潔的ASCII表格"""
    
    # 準備數據
    params = [
        ("Ideality Factor (n)", f"{result['n']:.3f}", ""),
        ("Schottky Barrier (Φ_b)", f"{result['phi_b']:.3f} eV", ""),
        ("Rs (Method A)", f"{result['Rs_a']:.1f} Ω", f"R²={result['r2_a']:.4f}"),
        ("Rs (Method B)", f"{result['Rs_b']:.1f} Ω", f"R²={result['r2_b']:.4f}"),
        ("Fit Range", f"{result['fit_range'][0]:.3f}-{result['fit_range'][1]:.3f} V", ""),
        ("Fit Quality", result['fit_quality'].upper(), f"{result['window_size']} pts")
    ]
    
    print("\n" + "=" * 50)
    print(" " * 10 + "CHEUNG METHOD ANALYSIS")
    print("=" * 50)
    
    for label, value, extra in params:
        print(f"  {label:<25} : {value:<15} {extra}")
    
    # 計算一致性
    rs_diff_percent = abs(result['Rs_a']-result['Rs_b'])/result['Rs_a']*100
    print(f"  {'Rs Consistency':<25} : {rs_diff_percent:.1f}% difference")
    
    print("-" * 50)
    print(f"  Analysis Method: {result['method']}")
    print(f"  Total Data Points: {len(result['bias'])}")
    print("=" * 50)

# 使用簡潔版本
#print_simple_table(result)

from tqdm import tqdm # 用來顯示進度條

def generate_schottky_maps(h5map, area_cm2):
    """
    計算全區域的 SBH, n, Rs Maps
    """
    bias_vec = h5map.bias_vector
    # 獲取正確的形狀
    if len(h5map.current_map.shape) == 2:
        # 形狀應該是 (bias_points, total_pixels)
        total_pixels = h5map.current_map.shape[1]
    elif len(h5map.current_map.shape) == 3:
        # 形狀可能是 (rows, cols, bias_points) 或 (bias_points, rows, cols)
        total_pixels = h5map.current_map.shape[1] * h5map.current_map.shape[2]
    else:
        raise ValueError(f"Unexpected current_map shape: {h5map.current_map.shape}")
    
    # 獲取像素尺寸
    xpixels = h5map.metadata.x_pixels
    ypixels = h5map.metadata.y_pixels
    
    # 檢查一致性
    if xpixels * ypixels != total_pixels:
        print(f"Warning: Metadata says {xpixels}x{ypixels}={xpixels*ypixels} pixels, but data has {total_pixels}")
        # 嘗試自動推斷
        xpixels = int(np.sqrt(total_pixels))
        ypixels = total_pixels // xpixels
    
    # 初始化空的 Map (使用 NaN)
    sbh_map = np.full(total_pixels, np.nan)
    n_map = np.full(total_pixels, np.nan)
    rs_map = np.full(total_pixels, np.nan)
    r2_map = np.full(total_pixels, np.nan)  # 新增：記錄擬合質量
    success_map = np.zeros(total_pixels, dtype=bool)  # 新增：記錄是否成功
    
    print(f"Starting Cheung Analysis for {total_pixels} pixels ({xpixels}x{ypixels})...")
    
    for i in tqdm(range(total_pixels), desc="Processing pixels"):
        # 提取該像素點的完整 I-V 曲線
        if len(h5map.current_map.shape) == 2:
            current_iv = h5map.current_map[:, i]
        elif len(h5map.current_map.shape) == 3:
            # 假設形狀是 (rows, cols, bias_points)
            row = i // h5map.current_map.shape[1]
            col = i % h5map.current_map.shape[1]
            current_iv = h5map.current_map[row, col, :]
        
        # 執行 Cheung 分析
        res = advanced_cheung_analysis(bias_vec, current_iv, area_cm2)
        
        if res['success']:
            sbh_map[i] = res['phi_b']
            n_map[i] = res['n']
            rs_map[i] = res['Rs_a']
            r2_map[i] = res['r2_a']
            success_map[i] = True
        else:
            # 可選：記錄失敗原因
            pass
    
    # Reshape 回 2D 矩陣
    sbh_2d = sbh_map.reshape(ypixels, xpixels)
    n_2d = n_map.reshape(ypixels, xpixels)
    rs_2d = rs_map.reshape(ypixels, xpixels)
    r2_2d = r2_map.reshape(ypixels, xpixels)
    success_2d = success_map.reshape(ypixels, xpixels)
    
    # 計算成功率
    success_rate = np.mean(success_map) * 100
    print(f"\nAnalysis completed. Success rate: {success_rate:.1f}%")
    
    return {
        'sbh': sbh_2d,
        'n': n_2d,
        'rs': rs_2d,
        'r2': r2_2d,
        'success': success_2d,
        'success_rate': success_rate,
        'shape': (ypixels, xpixels)
    }
    
def plot_analysis_maps(h5map, analysis_results, target_bias):
    """
    繪製多個分析圖
    """
    # 獲取電流圖 - 修正這裡的錯誤
    try:
        # 方法1: 使用h5map的方法獲取電流
        curr_1d = h5map.get_current_at_bias(target_bias)
        # 重塑為2D形狀
        curr_2d = h5map.reshape_to_2d(curr_1d)
    except (AttributeError, TypeError):
        # 方法2: 手動獲取和重塑
        # 找到最接近的偏壓點
        bias_vec = h5map.bias_vector
        idx = np.argmin(np.abs(bias_vec - target_bias))
        print(f"Using bias point {idx}: {bias_vec[idx]:.3f} V")
        
        # 獲取該偏壓下的電流
        if len(h5map.current_map.shape) == 2:
            # 形狀為 (bias_points, total_pixels)
            curr_1d = h5map.current_map[idx, :]
        elif len(h5map.current_map.shape) == 3:
            # 形狀為 (rows, cols, bias_points)
            curr_1d = h5map.current_map[:, :, idx].flatten()
        else:
            raise ValueError(f"Unexpected current_map shape: {h5map.current_map.shape}")
        
        # 使用分析結果的形狀進行重塑
        shape = analysis_results['shape']
        curr_2d = curr_1d.reshape(shape)
    
    # 獲取分析結果
    sbh_2d = analysis_results['sbh']
    n_2d = analysis_results['n']
    rs_2d = analysis_results['rs']
    r2_2d = analysis_results['r2']
    success_2d = analysis_results['success']
    
    # 創建圖表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 電流圖 (左上)
    # 檢查並確保 curr_2d 是正確的2D形狀
    if curr_2d.ndim != 2:
        print(f"Warning: curr_2d has shape {curr_2d.shape}, reshaping to {analysis_results['shape']}")
        curr_2d = curr_2d.reshape(analysis_results['shape'])
    
    im1 = axes[0, 0].imshow(np.abs(curr_2d), cmap='viridis', origin='lower')
    plt.colorbar(im1, ax=axes[0, 0], shrink = 0.3, aspect = 7)
    axes[0, 0].set_title(f'Current Map at {target_bias:.3f} V\n(Shape: {curr_2d.shape})')
    axes[0, 0].set_xlabel('X (pixels)')
    axes[0, 0].set_ylabel('Y (pixels)')
    
    # 2. SBH 圖 (中上)
    cmap_sbh = plt.cm.plasma.copy()
    cmap_sbh.set_bad(color='gray', alpha=0.3)
    
    # 只顯示成功的點
    sbh_to_plot = np.where(success_2d, sbh_2d, np.nan)
    
    # 計算顏色範圍（排除極端值）
    sbh_valid = sbh_to_plot[~np.isnan(sbh_to_plot)]
    if len(sbh_valid) > 0:
        vmin, vmax = np.percentile(sbh_valid, [5, 95])
    else:
        vmin, vmax = 0, 1
    
    im2 = axes[0, 1].imshow(sbh_to_plot, cmap=cmap_sbh, 
                           vmin=vmin, vmax=vmax, origin='lower')
    plt.colorbar(im2, ax=axes[0, 1], shrink = 0.3, aspect = 7)
    
    sbh_mean = np.nanmean(sbh_to_plot)
    sbh_std = np.nanstd(sbh_to_plot)
    axes[0, 1].set_title(f'Schottky Barrier Height\n(Φ_b = {sbh_mean:.3f} ± {sbh_std:.3f} eV)')
    axes[0, 1].set_xlabel('X (pixels)')
    axes[0, 1].set_ylabel('Y (pixels)')
    
    # 3. 理想因子圖 (右上)
    cmap_n = plt.cm.coolwarm.copy()
    cmap_n.set_bad(color='gray', alpha=0.3)
    
    n_to_plot = np.where(success_2d, n_2d, np.nan)
    # 計算合適的顯示範圍
    n_valid = n_to_plot[~np.isnan(n_to_plot)]
    if len(n_valid) > 0:
        n_vmin, n_vmax = np.percentile(n_valid, [5, 95])
        n_vmin = max(1, n_vmin * 0.8)  # 確保最小值至少為1
        n_vmax = min(10, n_vmax * 1.2)  # 限制最大值
    else:
        n_vmin, n_vmax = 1, 10
    
    im3 = axes[0, 2].imshow(n_to_plot, cmap=cmap_n, 
                           vmin=n_vmin, vmax=n_vmax, origin='lower')
    plt.colorbar(im3, ax=axes[0, 2], shrink = 0.3, aspect = 7)
    
    n_mean = np.nanmean(n_to_plot)
    n_median = np.nanmedian(n_to_plot)
    axes[0, 2].set_title(f'Ideality Factor\n(mean={n_mean:.2f}, median={n_median:.2f})')
    axes[0, 2].set_xlabel('X (pixels)')
    axes[0, 2].set_ylabel('Y (pixels)')
    
    # 4. 串聯電阻圖 (左下)
    cmap_rs = plt.cm.magma.copy()
    cmap_rs.set_bad(color='gray', alpha=0.3)
    
    rs_to_plot = np.where(success_2d, rs_2d, np.nan)
    
    # 只取正值的電阻
    rs_valid = rs_to_plot[~np.isnan(rs_to_plot) & (rs_to_plot > 0)]
    
    if len(rs_valid) > 0:
        # 使用對數尺度
        rs_valid_log = np.log10(rs_valid)
        vmin_r, vmax_r = np.percentile(rs_valid_log, [5, 95])
        
        # 創建對數標準化
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=10**vmin_r, vmax=10**vmax_r)
        
        im4 = axes[1, 0].imshow(rs_to_plot, cmap=cmap_rs, norm=norm, origin='lower')
        plt.colorbar(im4, ax=axes[1, 0], shrink = 0.3, aspect = 7)
    else:
        im4 = axes[1, 0].imshow(rs_to_plot, cmap=cmap_rs, origin='lower')
        plt.colorbar(im4, ax=axes[1, 0], shrink = 0.3, aspect = 7)
    
    rs_median = np.nanmedian(rs_to_plot)
    axes[1, 0].set_title(f'Series Resistance\n(median={rs_median:.1f} Ω)')
    axes[1, 0].set_xlabel('X (pixels)')
    axes[1, 0].set_ylabel('Y (pixels)')
    
    # 5. R² 擬合質量圖 (中下)
    cmap_r2 = plt.cm.RdYlGn.copy()
    cmap_r2.set_bad(color='gray', alpha=0.3)
    
    r2_to_plot = np.where(success_2d, r2_2d, np.nan)
    im5 = axes[1, 1].imshow(r2_to_plot, cmap=cmap_r2, 
                           vmin=0.9, vmax=1.0, origin='lower')
    plt.colorbar(im5, ax=axes[1, 1], shrink = 0.3, aspect = 7)
    
    r2_mean = np.nanmean(r2_to_plot)
    axes[1, 1].set_title(f'Fit Quality (R²)\n(mean={r2_mean:.4f})')
    axes[1, 1].set_xlabel('X (pixels)')
    axes[1, 1].set_ylabel('Y (pixels)')
    
    # 6. 成功率/統計圖 (右下)
    axes[1, 2].axis('off')
    
    # 統計信息
    total_pixels = success_2d.size
    success_count = np.sum(success_2d)
    success_rate = analysis_results['success_rate']
    
    # 計算各種統計量
    sbh_stats = f"{np.nanmean(sbh_to_plot):.4f} ± {np.nanstd(sbh_to_plot):.4f} eV"
    n_stats = f"{np.nanmean(n_to_plot):.3f} ± {np.nanstd(n_to_plot):.3f}"
    rs_stats = f"{np.nanmedian(rs_to_plot):.0f} Ω"
    
    stats_text = (
        f"ANALYSIS STATISTICS\n"
        f"═══════════════════\n\n"
        f"Total Pixels: {total_pixels}\n"
        f"Successful Fits: {success_count} ({success_rate:.1f}%)\n"
        f"Failed Fits: {total_pixels - success_count}\n\n"
        
        f"SBH (Φ_b) STATISTICS\n"
        f"═══════════════════\n"
        f"Mean ± Std: {sbh_stats}\n"
        f"Min: {np.nanmin(sbh_to_plot):.4f} eV\n"
        f"Max: {np.nanmax(sbh_to_plot):.4f} eV\n"
        f"Median: {np.nanmedian(sbh_to_plot):.4f} eV\n\n"
        
        f"IDEALITY FACTOR STATS\n"
        f"═════════════════════\n"
        f"Mean ± Std: {n_stats}\n"
        f"Median: {np.nanmedian(n_to_plot):.3f}\n"
        f"Range: {np.nanmin(n_to_plot):.2f}-{np.nanmax(n_to_plot):.2f}\n\n"
        
        f"SERIES RESISTANCE\n"
        f"═════════════════\n"
        f"Median: {rs_stats}\n"
        f"Mean: {np.nanmean(rs_to_plot):.0f} Ω\n"
        f"Range: {np.nanmin(rs_to_plot):.0f}-{np.nanmax(rs_to_plot):.0f} Ω\n\n"
        
        f"FIT QUALITY\n"
        f"═══════════\n"
        f"Mean R²: {r2_mean:.4f}\n"
        f"R² > 0.99: {np.sum(r2_to_plot > 0.99)} pixels\n"
        f"R² > 0.95: {np.sum(r2_to_plot > 0.95)} pixels"
    )
    
    axes[1, 2].text(0.05, 0.95, stats_text, 
                   fontfamily='monospace', fontsize=9,
                   verticalalignment='top', transform=axes[1, 2].transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return fig, axes



import os
from datetime import datetime

def setup_output_directory(base_dir='schottky-cheung', create_subdir=True):
    """
    設置輸出目錄結構
    
    參數:
    ----------
    base_dir : str
        基礎目錄名稱
    create_subdir : bool
        是否創建帶時間戳的子目錄
        
    返回:
    ----------
    output_dir : str
        完整的輸出目錄路徑
    """
    # 確保基礎目錄存在
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created directory: {base_dir}")
    
    if create_subdir:
        # 創建帶時間戳的子目錄
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(base_dir, f"analysis_{timestamp}")
    else:
        output_dir = base_dir
    
    # 創建輸出目錄
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # 在目錄中創建子目錄結構
    subdirs = ['figures', 'data', 'logs', 'reports']
    for subdir in subdirs:
        subdir_path = os.path.join(output_dir, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
            print(f"Created subdirectory: {subdir_path}")
    
    return output_dir

def save_single_pixel_analysis(bias, current, result, xs, ys, output_dir='schottky-cheung'):
    """
    保存單個像素點的 Cheung 分析結果
    
    參數:
    ----------
    bias : array
        偏壓陣列
    current : array
        電流陣列
    result : dict
        分析結果
    xs, ys : int
        像素座標
    output_dir : str
        輸出目錄
        
    返回:
    ----------
    saved_paths : dict
        保存的文件路徑
    """
    # 設置輸出目錄
    output_dir = setup_output_directory(output_dir, create_subdir=False)
    
    # 創建像素點專用子目錄
    pixel_dir = os.path.join(output_dir, 'figures', f'pixel_{xs}_{ys}')
    if not os.path.exists(pixel_dir):
        os.makedirs(pixel_dir)
    
    # 生成文件名前綴
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_prefix = f"{timestamp}_pixel_{xs}_{ys}"
    
    saved_paths = {}
    
    # 1. 保存圖表
    fig = plot_cheung_results(bias, current, result)
    if fig is not None:
        # 高分辨率版本
        high_res_path = os.path.join(pixel_dir, f"{filename_prefix}_highres.png")
        fig.savefig(high_res_path, dpi=300, bbox_inches='tight')
        saved_paths['figure_highres'] = high_res_path
        
        # 中等分辨率版本
        medium_res_path = os.path.join(pixel_dir, f"{filename_prefix}.png")
        fig.savefig(medium_res_path, dpi=150, bbox_inches='tight')
        saved_paths['figure'] = medium_res_path
        
        # PDF 版本 (矢量圖)
        pdf_path = os.path.join(pixel_dir, f"{filename_prefix}.pdf")
        fig.savefig(pdf_path, bbox_inches='tight')
        saved_paths['figure_pdf'] = pdf_path
        
        plt.close(fig)
        print(f"圖表已保存到: {pixel_dir}")
    
    # 2. 保存數據為 CSV
    data_dir = os.path.join(output_dir, 'data')
    csv_path = os.path.join(data_dir, f"{filename_prefix}_data.csv")
    
    # 準備數據
    data_dict = {
        'bias_V': bias,
        'current_A': current,
        'I_smooth_A': result['I_smooth'],
        'y_a_V': result['y_a'],
        'H_I_V': result['H_I']
    }
    
    # 轉換為DataFrame並保存
    try:
        import pandas as pd
        df = pd.DataFrame(data_dict)
        df.to_csv(csv_path, index=False)
        saved_paths['data_csv'] = csv_path
        print(f"數據已保存到: {csv_path}")
    except ImportError:
        print("Pandas 未安裝，跳過 CSV 保存")
    
    # 3. 保存分析結果為 JSON
    import json
    results_path = os.path.join(data_dir, f"{filename_prefix}_results.json")
    
    # 準備可序列化的結果字典
    results_to_save = {}
    for key, value in result.items():
        if isinstance(value, np.ndarray):
            results_to_save[key] = value.tolist()  # 轉換為列表
        elif isinstance(value, np.generic):
            results_to_save[key] = value.item()  # 轉換為Python標量
        else:
            results_to_save[key] = value
    
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    saved_paths['results_json'] = results_path
    print(f"分析結果已保存到: {results_path}")
    
    # 4. 保存摘要報告為文本文件
    report_dir = os.path.join(output_dir, 'reports')
    report_path = os.path.join(report_dir, f"{filename_prefix}_summary.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"CHEUNG METHOD ANALYSIS REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Pixel Coordinates: ({xs}, {ys})\n")
        f.write(f"Total Data Points: {len(bias)}\n")
        f.write(f"Bias Range: {bias[0]:.3f} to {bias[-1]:.3f} V\n\n")
        
        f.write("EXTRACTED PARAMETERS\n")
        f.write("-"*60 + "\n")
        f.write(f"Ideality Factor (n): {result['n']:.3f}\n")
        f.write(f"Schottky Barrier Height (Φ_b): {result['phi_b']:.3f} eV\n")
        f.write(f"Series Resistance Rs (Method A): {result['Rs_a']:.1f} Ω\n")
        f.write(f"Series Resistance Rs (Method B): {result['Rs_b']:.1f} Ω\n")
        f.write(f"Rs Difference: {abs(result['Rs_a']-result['Rs_b']):.1f} Ω ({abs(result['Rs_a']-result['Rs_b'])/result['Rs_a']*100:.1f}%)\n\n")
        
        f.write("GOODNESS OF FIT\n")
        f.write("-"*60 + "\n")
        f.write(f"R² (Method A): {result['r2_a']:.4f}\n")
        f.write(f"R² (Method B): {result['r2_b']:.4f}\n\n")
        
        f.write("FIT REGION INFORMATION\n")
        f.write("-"*60 + "\n")
        f.write(f"Fit Range: {result['fit_range'][0]:.3f} to {result['fit_range'][1]:.3f} V\n")
        f.write(f"Window Size: {result['window_size']} points\n")
        f.write(f"Fit Quality: {result['fit_quality'].upper()}\n\n")
        
        f.write("ADDITIONAL INFORMATION\n")
        f.write("-"*60 + "\n")
        f.write(f"Temperature: 300 K\n")
        f.write(f"Contact Area: {np.pi*(12.5e-7)**2:.2e} cm²\n")
        f.write(f"Analysis Method: {result['method']}\n")
        f.write(f"Success: {result['success']}\n")
    
    saved_paths['report_txt'] = report_path
    print(f"摘要報告已保存到: {report_path}")
    
    # 5. 保存日誌
    log_dir = os.path.join(output_dir, 'logs')
    log_path = os.path.join(log_dir, f"{timestamp}_analysis.log")
    
    with open(log_path, 'a') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ")
        f.write(f"Pixel ({xs}, {ys}) analysis completed. ")
        f.write(f"n={result['n']:.3f}, Φ_b={result['phi_b']:.3f}eV, ")
        f.write(f"R²_A={result['r2_a']:.4f}, R²_B={result['r2_b']:.4f}\n")
    
    saved_paths['log'] = log_path
    
    print("\n" + "="*60)
    print("所有文件已保存到以下路徑:")
    for key, path in saved_paths.items():
        print(f"  {key}: {path}")
    print("="*60)
    
    return saved_paths

def save_full_map_analysis(h5map, analysis_results, target_bias, output_dir='schottky-cheung'):
    """
    保存全圖分析結果
    
    參數:
    ----------
    h5map : object
        HDF5地圖數據對象
    analysis_results : dict
        分析結果
    target_bias : float
        目標偏壓值
    output_dir : str
        輸出目錄
        
    返回:
    ----------
    saved_paths : dict
        保存的文件路徑
    """
    # 設置輸出目錄（創建帶時間戳的子目錄）
    output_dir = setup_output_directory(output_dir, create_subdir=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_prefix = f"{timestamp}_fullmap"
    
    saved_paths = {}
    
    # 1. 繪製並保存圖表
    fig, axes = plot_analysis_maps(h5map, analysis_results, target_bias)
    
    # 保存多個版本的圖表
    figures_dir = os.path.join(output_dir, 'figures')
    
    # 高分辨率PNG
    highres_path = os.path.join(figures_dir, f"{filename_prefix}_highres.png")
    fig.savefig(highres_path, dpi=300, bbox_inches='tight')
    saved_paths['figure_highres'] = highres_path
    
    # 中等分辨率PNG
    norm_path = os.path.join(figures_dir, f"{filename_prefix}.png")
    fig.savefig(norm_path, dpi=150, bbox_inches='tight')
    saved_paths['figure'] = norm_path
    
    # PDF版本
    pdf_path = os.path.join(figures_dir, f"{filename_prefix}.pdf")
    fig.savefig(pdf_path, bbox_inches='tight')
    saved_paths['figure_pdf'] = pdf_path
    
    plt.close(fig)
    print(f"全圖分析圖表已保存到: {figures_dir}")
    
    # 2. 保存數據為NPZ文件
    data_dir = os.path.join(output_dir, 'data')
    npz_path = os.path.join(data_dir, f"{filename_prefix}_data.npz")
    
    np.savez_compressed(
        npz_path,
        sbh_map=analysis_results['sbh'],
        n_map=analysis_results['n'],
        rs_map=analysis_results['rs'],
        r2_map=analysis_results['r2'],
        success_map=analysis_results['success'],
        success_rate=analysis_results['success_rate'],
        shape=analysis_results['shape'],
        metadata={
            'timestamp': timestamp,
            'target_bias': target_bias,
            'total_pixels': analysis_results['success'].size,
            'successful_pixels': np.sum(analysis_results['success']),
            'analysis_method': 'Cheung Method'
        }
    )
    saved_paths['data_npz'] = npz_path
    print(f"數據已保存到: {npz_path}")
    
    # 3. 保存統計報告
    report_dir = os.path.join(output_dir, 'reports')
    report_path = os.path.join(report_dir, f"{filename_prefix}_statistics.txt")
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("FULL MAP CHEUNG ANALYSIS STATISTICS REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Target Bias: {target_bias:.3f} V\n")
        f.write(f"Map Shape: {analysis_results['shape']} (Y×X)\n")
        f.write(f"Total Pixels: {analysis_results['success'].size}\n\n")
        
        # 成功率統計
        success_count = np.sum(analysis_results['success'])
        success_rate = analysis_results['success_rate']
        f.write("SUCCESS RATE STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Successful Fits: {success_count}\n")
        f.write(f"Failed Fits: {analysis_results['success'].size - success_count}\n")
        f.write(f"Success Rate: {success_rate:.1f}%\n\n")
        
        # SBH統計
        sbh_valid = analysis_results['sbh'][analysis_results['success']]
        f.write("SCHOTTKY BARRIER HEIGHT (Φ_b) STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Mean: {np.mean(sbh_valid):.4f} eV\n")
        f.write(f"Std Dev: {np.std(sbh_valid):.4f} eV\n")
        f.write(f"Median: {np.median(sbh_valid):.4f} eV\n")
        f.write(f"Min: {np.min(sbh_valid):.4f} eV\n")
        f.write(f"Max: {np.max(sbh_valid):.4f} eV\n")
        f.write(f"25th Percentile: {np.percentile(sbh_valid, 25):.4f} eV\n")
        f.write(f"75th Percentile: {np.percentile(sbh_valid, 75):.4f} eV\n\n")
        
        # 理想因子統計
        n_valid = analysis_results['n'][analysis_results['success']]
        f.write("IDEALITY FACTOR (n) STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Mean: {np.mean(n_valid):.3f}\n")
        f.write(f"Std Dev: {np.std(n_valid):.3f}\n")
        f.write(f"Median: {np.median(n_valid):.3f}\n")
        f.write(f"Min: {np.min(n_valid):.2f}\n")
        f.write(f"Max: {np.max(n_valid):.2f}\n\n")
        
        # 串聯電阻統計
        rs_valid = analysis_results['rs'][analysis_results['success']]
        rs_valid = rs_valid[rs_valid > 0]  # 只取正值
        if len(rs_valid) > 0:
            f.write("SERIES RESISTANCE (R_s) STATISTICS\n")
            f.write("-"*70 + "\n")
            f.write(f"Median: {np.median(rs_valid):.0f} Ω\n")
            f.write(f"Mean: {np.mean(rs_valid):.0f} Ω\n")
            f.write(f"Std Dev: {np.std(rs_valid):.0f} Ω\n")
            f.write(f"Min: {np.min(rs_valid):.0f} Ω\n")
            f.write(f"Max: {np.max(rs_valid):.0f} Ω\n\n")
        
        # 擬合質量統計
        r2_valid = analysis_results['r2'][analysis_results['success']]
        f.write("FIT QUALITY STATISTICS\n")
        f.write("-"*70 + "\n")
        f.write(f"Mean R²: {np.mean(r2_valid):.4f}\n")
        f.write(f"Std Dev R²: {np.std(r2_valid):.4f}\n")
        f.write(f"R² > 0.99: {np.sum(r2_valid > 0.99)} pixels\n")
        f.write(f"R² > 0.95: {np.sum(r2_valid > 0.95)} pixels\n")
        f.write(f"R² > 0.90: {np.sum(r2_valid > 0.90)} pixels\n\n")
        
        f.write("ANALYSIS PARAMETERS\n")
        f.write("-"*70 + "\n")
        f.write(f"Contact Diameter: 25 nm\n")
        f.write(f"Contact Area: {np.pi*(12.5e-7)**2:.2e} cm²\n")
        f.write(f"Temperature: 300 K\n")
        f.write(f"Analysis Method: Cheung Method with sliding window\n")
    
    saved_paths['report_txt'] = report_path
    print(f"統計報告已保存到: {report_path}")
    
    # 4. 保存日誌
    log_dir = os.path.join(output_dir, 'logs')
    log_path = os.path.join(log_dir, f"{timestamp}_fullmap.log")
    
    with open(log_path, 'a') as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ")
        f.write(f"Full map analysis completed.\n")
        f.write(f"  Map shape: {analysis_results['shape']}\n")
        f.write(f"  Success rate: {success_rate:.1f}%\n")
        f.write(f"  Mean SBH: {np.mean(sbh_valid):.4f} eV\n")
        f.write(f"  Mean ideality factor: {np.mean(n_valid):.3f}\n")
        f.write(f"  Files saved:\n")
        for key, path in saved_paths.items():
            f.write(f"    {key}: {os.path.basename(path)}\n")
    
    saved_paths['log'] = log_path
    
    # 5. 創建README文件
    readme_path = os.path.join(output_dir, "README.txt")
    with open(readme_path, 'w') as f:
        f.write("Schottky Barrier Analysis - Cheung Method Results\n")
        f.write("="*60 + "\n\n")
        f.write("Directory Structure:\n")
        f.write("  figures/     - 圖表文件 (PNG, PDF)\n")
        f.write("  data/        - 原始數據和分析結果\n")
        f.write("  reports/     - 分析報告和統計\n")
        f.write("  logs/        - 分析日誌\n\n")
        f.write(f"Analysis performed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Map dimensions: {analysis_results['shape']} pixels\n")
        f.write(f"Success rate: {success_rate:.1f}%\n\n")
        f.write("Files included:\n")
        for key, path in saved_paths.items():
            f.write(f"  {os.path.basename(path)} - {key}\n")
    
    print("\n" + "="*60)
    print("全圖分析已完成，所有文件已保存到以下目錄:")
    print(f"  {output_dir}")
    print("\n保存的文件:")
    for key, path in saved_paths.items():
        print(f"  {key}: {os.path.relpath(path, output_dir)}")
    print("="*60)
    
    return saved_paths, output_dir
    
# older version   
def plot_cheung_results_with_deviation(bias, current, res, pixel_coord=(0,0)):
    if res is None or len(res['indices']) == 0:
        print(f"Skipping plot for {pixel_coord}: No valid fit indices.")
        return # 直接結束，避免執行後面的 np.min()

    # 1. 準備數據
    I = np.abs(current)
    y_a = res['y_a']
    H_I = res['H_I']
    v_start, v_end = res['fit_range']
    fit_indices = res['indices']
    
    # 建立畫布
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(wspace=0.3, bottom=0.15)
    fig.suptitle(fr"Cheung Method Analysis at Pixel {pixel_coord}", fontsize=17, y=1.02)

    # --- Plot 1: Semi-log I-V ---
    ax0 = axes[0, 0]
    ax0.semilogy(bias, I, color='gray', lw = 3, alpha=0.6, label='Full Data')
    ax0.semilogy(bias[fit_indices], I[fit_indices], 'r.', label='Fitting Range')
    ax0.axvspan(v_start, v_end, color='green', alpha=0.1)
    ax0.set_xlabel('Voltage (V)', fontsize = 15)
    ax0.set_ylabel('Current (A)', fontsize = 15)
    ax0.set_title('I-V Characteristic (logy)', fontsize = 15)
    ax0.grid(True, linestyle='-', alpha=0.4)
    ax0.legend(fontsize = 11)

    # --- Plot 2 normal I-V
    ax1 = axes[0, 1]
    ax1.plot(bias, I, color='gray', lw = 3, alpha=0.6, label='Full Data')
    ax1.plot(bias[fit_indices], I[fit_indices], 'r.', label='Fitting Range')
    ax1.axvspan(v_start, v_end, color='green', alpha=0.1)
    ax1.set_xlabel('Voltage (V)', fontsize = 15)
    ax1.set_ylabel('Current (A)', fontsize = 15)
    ax1.set_title('I-V Characteristic', fontsize = 15)
    ax1.grid(True, linestyle='-', alpha=0.4)
    ax1.legend(fontsize = 11)

    # --- Plot 2: Method A (dV/dlnI vs I) ---
    ax2 = axes[1, 0]
    # 畫出全部點，看整體的彎曲趨勢
    ax2.scatter(I, y_a, c='gray', s=20, alpha=0.2, label='Outside Range')
    # 畫出擬合點
    ax2.scatter(I[fit_indices], y_a[fit_indices], c='red', s=30, label='Fit Range')
    
    # 繪製擬合線 (延伸一點點以便觀察)
    I_range = I[fit_indices]
    I_ext = np.linspace(np.min(I_range)*0.5, np.max(I_range)*1.2, 100)
    V_T = 8.617333e-5 * 300 # 假設 T=300K
    intercept_a = res['n'] * V_T
    ax2.plot(I_ext, res['Rs_a'] * I_ext + intercept_a, 'k--', alpha=0.8)
    
    ax2.set_xlabel('Current I (A)', fontsize = 15)
    ax2.set_ylabel('dV / d(ln I)', fontsize = 15)
    ax2.set_title(r'Method A: $dV/d(\ln I) = nV_T + IR_s$', fontsize = 15)
    
    text_a = fr"Rs = {res['Rs_a']:.1f} $\Omega$" + "\n" + fr"$n$ = {res['n']:.2f}" + "\n" + fr"$R^2$ = {res['r2_a']:.4f}"
    ax2.text(0.05, 0.95, text_a, transform=ax2.transAxes, verticalalignment='top',
             fontsize = 12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # --- Plot 3: Method B (H(I) vs I) ---
    ax3 = axes[1, 1]
    # 畫出全部點
    ax3.scatter(I, H_I, c='gray', s=20, alpha=0.4, label='Outside Range')
    # 畫出擬合點
    ax3.scatter(I[fit_indices], H_I[fit_indices], c='blue', s=30, label='Fit Range')
    
    # 繪製擬合線
    intercept_b = res['phi_b'] * res['n']
    ax3.plot(I_ext, res['Rs_b'] * I_ext + intercept_b, 'k--', alpha=0.8)
    
    ax3.set_xlabel('Current I (A)',fontsize = 15)
    ax3.set_ylabel('H(I)', fontsize = 15)
    ax3.set_title(r'Method B: $H(I) = n\Phi_b + IR_s$',fontsize = 15)
    
    text_b = fr"Rs = {res['Rs_b']:.1f} $\Omega$" + "\n" + fr"$\Phi_b$ = {res['phi_b']:.3f} eV" + "\n" + fr"$R^2$ = {res['r2_b']:.4f}"
    ax3.text(0.05, 0.95, text_b, transform=ax3.transAxes, verticalalignment='top',
             fontsize = 12,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    for ax, y_data in zip([ax2, ax3], [y_a, H_I]): # 直接配對 ax 與對應的數據
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, linestyle='-', alpha=0.4)
        
        # X 軸：以擬合範圍為主
        current_I_fit = I[fit_indices]
        ax.set_xlim(0, np.max(current_I_fit) * 1.4)
        
        # Y 軸：根據該 ax 專屬的擬合數據來決定範圍
        y_fit_vals = y_data[fit_indices]
        y_min, y_max = np.min(y_fit_vals), np.max(y_fit_vals)
        y_range = y_max - y_min
        
        # 給予上下 20% 的緩衝空間，避免數據貼邊
        ax.set_ylim(y_min - 0.2 * y_range, y_max + 0.2 * y_range)

    plt.tight_layout()
    
    from pathlib import Path

    # 設定儲存路徑
    save_dir = Path("analysis_results/schottky_plots")
    save_dir.mkdir(parents=True, exist_ok=True) # 自動建立資料夾
    
    # 檔名包含座標，方便溯源
    file_name = f"point_{pixel_coord}_cheung_sliding_window_fit.png"
    save_path = save_dir / file_name

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"圖表已儲存至: {save_path}")
    
    plt.show()