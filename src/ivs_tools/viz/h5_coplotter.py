from typing import List, Tuple, Dict, Optional, Union
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

from ivs_tools import HDF5Analyzer

class CorrelatedPlotter:
    """創建顏色對應的熱圖和I-V曲線圖"""

    def __init__(self, h5_analyzer: HDF5Analyzer):
        self.h5 = h5_analyzer
        self.rois: Dict[str, Dict] = {} 
        # 定義預設顏色方案
        self._default_colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#aec7e8', '#ffbb78', '#98df8a', '#ff9896', '#c5b0d5',
            '#c49c94', '#f7b6d2', '#c7c7c7', '#dbdb8d', '#9edae5'
        ]
        
    def _get_point_colors(self, num_points: int, colormap: str = "tab10") -> List:
        """
        獲取點的顏色
        
        Args:
            num_points: 點的數量
            colormap: 色彩映射名稱
            
        Returns:
            顏色列表
        """
        # 如果點數少於等於預設顏色數，使用預設顏色
        if num_points <= len(self._default_colors):
            return self._default_colors[:num_points]
        
        # 否則使用指定的 colormap
        try:
            cmap = plt.get_cmap(colormap)
            return [cmap(i / max(num_points - 1, 1)) for i in range(num_points)]
        except:
            # 如果 colormap 無效，使用 matplotlib 預設顏色循環
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            return [color_cycle[i % len(color_cycle)] for i in range(num_points)]
            
    def add_roi(self, name: str, bias_value: float, 
                x_range: Tuple[int, int], y_range: Tuple[int, int]) -> np.ndarray:
        """
        添加 ROI
        
        Args:
            name: ROI 名稱
            bias_value: bias 值
            x_range: (x_start, x_end) 像素坐標
            y_range: (y_start, y_end) 像素坐標
            
        Returns:
            ROI 數據
        """
        current_2d = self.h5.get_2d_map_at_bias(bias_value)
        x_start, x_end = x_range
        y_start, y_end = y_range
        
        # 確保範圍在邊界內
        height, width = current_2d.shape
        x_start = max(0, x_start)
        x_end = min(width, x_end)
        y_start = max(0, y_start)
        y_end = min(height, y_end)
        
        roi_data = current_2d[y_start:y_end, x_start:x_end]
        
        self.rois[name] = {
            'data': roi_data,
            'coords': (x_start, x_end, y_start, y_end),
            'bias': bias_value,
            'shape': roi_data.shape
        }
        
        return roi_data
    
    def remove_roi(self, name: str) -> bool:
        """移除 ROI"""
        if name in self.rois:
            del self.rois[name]
            return True
        return False
    
    def get_roi(self, name: str) -> Optional[Dict]:
        """獲取 ROI 資訊"""
        return self.rois.get(name)
    
    def get_roi_statistics(self, name: str) -> Dict[str, float]:
        """獲取 ROI 統計數據"""
        if name not in self.rois:
            raise KeyError(f"ROI '{name}' not found")
        
        roi_data = self.rois[name]['data']
        
        return {
            'name': name,
            'shape': roi_data.shape,
            'size': roi_data.size,
            'mean': float(np.mean(roi_data)),
            'std': float(np.std(roi_data)),
            'min': float(np.min(roi_data)),
            'max': float(np.max(roi_data)),
            'median': float(np.median(roi_data)),
            'q25': float(np.percentile(roi_data, 25)),
            'q75': float(np.percentile(roi_data, 75)),
            'bias': self.rois[name]['bias']
        }

     # ========== 擴展的 create_correlated_plot 方法 ==========
    def create_correlated_plot_with_roi(self, 
                                        bias_value: float,
                                        points: List[Tuple[int, int]],
                                        roi_coords: Tuple[int, int, int, int],  # (x_start, x_end, y_start, y_end)
                                        islogy: bool = False,
                                        start_idx: int = 3,
                                        figsize: Tuple[int, int] = (25, 6),
                                        colormap: str = "tab10"):
        """
        創建包含 ROI 熱圖的三圖
        
        Args:
            bias_value: 熱圖顯示的bias值
            points: 點列表 [(x1, y1), (x2, y2), ...]
            roi_coords: ROI 座標 (x_start, x_end, y_start, y_end)
            islogy: 是否使用對數y軸
            start_idx: 從第幾個bias點開始繪製
            figsize: 圖表大小
            colormap: 使用的色彩映射
        """

        fig, axes = plt.subplots(
            1, 3,
            figsize=figsize,
            gridspec_kw={"width_ratios": [1, 1, 1.6]}
        )
        
        metadata = self.h5.metadata
        pixel_size = metadata.x_range / metadata.x_pixels
        
        point_colors = self._get_point_colors(len(points), colormap)
        
        # ===== 圖1: 完整熱圖 =====
        current_2d_map = self.h5.get_2d_map_at_bias(bias_value)
        
        im1 = axes[0].imshow(current_2d_map * 1e6,
                            aspect='equal',
                            origin='lower',
                            cmap='Blues_r',
                            interpolation='nearest')
        
        # 繪製 scatter 點
        for i, (x, y) in enumerate(points):
            color = point_colors[i]
            axes[0].scatter(x, y, s=80, color=color,
                           edgecolors='white', linewidth=1.5, zorder=10)
        
        # 繪製 ROI 框
        x_start, x_end, y_start, y_end = roi_coords
        from matplotlib.patches import Rectangle
        rect = Rectangle((x_start, y_start), 
                        x_end - x_start, 
                        y_end - y_start,
                        fill=False, 
                        edgecolor='red',
                        linewidth=2,
                        linestyle='--')
        axes[0].add_patch(rect)
               
        axes[0].set_xticks([])
        axes[0].set_yticks([])
        axes[0].set_title(f"Bias = {bias_value} V", fontsize=16)
        cbar1 = plt.colorbar(im1, ax=axes[0], pad=0, aspect=9, shrink=0.3, anchor=(0.35, 0.6))
        cbar1.set_label('Current (μA)', fontsize=12)
        
        # 添加 scalebar
        pixel_size_nm = pixel_size * 1e9
        scalebar = ScaleBar(pixel_size_nm, 'nm',
                           length_fraction=0.25, height_fraction = 0.025,
                           location='lower right',
                           box_alpha=0.5,
                           color='black',
                           font_properties={'size': 10})
        axes[0].add_artist(scalebar)
        
        # ===== 圖2: ROI 熱圖 =====
        roi_data = current_2d_map[y_start:y_end, x_start:x_end]
        
        im2 = axes[1].imshow(roi_data * 1e6,
                            aspect='equal',
                            origin='lower',
                            cmap='Blues_r',
                            extent=[0, roi_data.shape[1], 0, roi_data.shape[0]])
        axes[1].set_xticks([])
        axes[1].set_yticks([])
        # 在 ROI 中標記點（如果點在 ROI 內）
        roi_points = []
        for i, (x, y) in enumerate(points):
            if x_start <= x < x_end and y_start <= y < y_end:
                roi_x = x - x_start
                roi_y = y - y_start
                color = point_colors[i]
                axes[1].scatter(roi_x, roi_y, s=80, color=color,
                               edgecolors='white', linewidth=1.5, zorder=10)

                roi_points.append((i, roi_x, roi_y, color))
        
        roi_size = f"{roi_data.shape[1]}×{roi_data.shape[0]} pixels"
        axes[1].set_title(f"ROI Zoom\n{roi_size}", fontsize=16)
        cbar2 = plt.colorbar(im2, ax=axes[1], pad=0, aspect=9, shrink=0.3, anchor=(0.35, 0.6))
        cbar2.set_label('Current (μA)', fontsize=12)
               
        # ===== 圖3: I-V 曲線 =====
        bias_vector = self.h5.bias_vector
        
        for i, (x, y) in enumerate(points):
            spectrum = self.h5.get_spectrum_at_pixel(x, y)
            color = point_colors[i]
            
            if islogy:
                axes[2].plot(bias_vector[start_idx:],
                           np.abs(spectrum[start_idx:]),
                           color=color,
                           lw=2,
                           label=f'P{i+1}: ({x}, {y})')
            else:
                axes[2].plot(bias_vector[start_idx:],
                           spectrum[start_idx:],
                           color=color,
                           lw=2,
                           label=f'P{i+1}: ({x}, {y})')
        
        axes[2].set_xlabel("Bias (V)", fontsize=18)
        axes[2].set_ylabel("Current (A)", fontsize=18)
        
        if islogy:
            axes[2].set_yscale('log')
        else:
            axes[2].ticklabel_format(axis='y', style='sci', scilimits=(-3, 3))
        
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(
            fontsize=14,  # 字體大小
            loc='lower right',  # 位置
            bbox_to_anchor=(0.96, 0.1),  #
            frameon=True,
            fancybox=True,
            shadow=True,
            title='',  # 圖例標題
            title_fontsize=12,
            ncol=2,  # 分為2欄
            columnspacing=0.8,  # 欄間距
            handlelength=1.5,  # 線條長度
            handletextpad=0.5,  # 線條與文字間距
            borderaxespad=0.5  # 邊框與座標軸間距
        )

        axes[2].tick_params(labelsize=14)
        
        plt.tight_layout()
        return fig, point_colors, roi_data
        
    def create_correlated_plot(self, 
                               bias_value: float,
                               points: List[Tuple[int, int]],
                               islogy: bool = False,
                               start_idx: int = 3,
                               figsize: Tuple[int, int] = (17, 6),
                               colormap: str = "tab10"):
        """
        創建顏色對應的雙圖
        """
        fig, ax = plt.subplots(1, 2, figsize=figsize)

        metadata = self.h5.metadata
        pixel_size = metadata.x_range / metadata.x_pixels

        num_points = len(points)
        
        point_colors = self._get_point_colors(len(points), colormap)
        
        current_2d_map = self.h5.get_2d_map_at_bias(bias_value)

        im = ax[0].imshow(current_2d_map * 1e6,
                         aspect="equal",
                         origin="lower",
                         cmap="Blues_r",
                         interpolation="nearest")

        cbar = fig.colorbar(im, ax=ax[0], orientation="horizontal",
                           pad=0, aspect=9, shrink=0.2, anchor=(0.35, 0.6))
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label("Current (μA)", fontsize=16)

        for i, (x, y) in enumerate(points):
            color = point_colors[i]

            ax[0].scatter(x, y, s=60,
                         color=color,
                         edgecolors="white",
                         linewidths=1,
                         zorder=11)

        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[0].set_title(f"Bias = {bias_value} V", fontsize=18, pad=15)

        pixel_size_nm = pixel_size * 1e9
        scalebar = ScaleBar(pixel_size_nm, "nm",
                           length_fraction=0.25, height_fraction = 0.025,
                           location="lower right",
                           box_alpha=0.8,
                           color="black",
                           font_properties={"size": 14})
        ax[0].add_artist(scalebar)

        bias_vector = self.h5.bias_vector

        for i, (x, y) in enumerate(points):
            spectrum = self.h5.get_spectrum_at_pixel(x, y)
            color = point_colors[i]

            if islogy:
                ax[1].plot(bias_vector[start_idx:],
                          np.abs(spectrum[start_idx:]),
                          color=color,
                          lw=2.5,
                          label=f"P{i+1}: ({x}, {y})")
            else:
                ax[1].plot(bias_vector[start_idx:],
                          spectrum[start_idx:],
                          color=color,
                          lw=2.5,
                          label=f"P{i+1}: ({x}, {y})"
                          )

        ax[1].set_xlabel("Bias (V)", fontsize=20)
        ax[1].set_ylabel("Current (A)", fontsize=20)

        if islogy:
            ax[1].set_yscale("log")
        else:
            ax[1].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
            ax[1].yaxis.get_offset_text().set_fontsize(14)

        ax[1].tick_params(axis="both", labelsize=14)
        ax[1].grid(True, alpha=0.2, linestyle="-")

        legend = ax[1].legend(fontsize=14,
                     loc='upper left',
                     bbox_to_anchor=(0.68, 0.34) if islogy else (0.68, 0.34),
                     frameon=True,
                     fancybox=True,
                     shadow=True,
                     framealpha=0.9)

        for i, text in enumerate(legend.get_texts()):
            text.set_color(point_colors[i])

        #ax[1].axvline(x=bias_value, color="red", linestyle="--", alpha=0.5, linewidth=1)
        #ax[1].text(bias_value + 0.05, ax[1].get_ylim()[0] + 0.1 * (ax[1].get_ylim()[1] - ax[1].get_ylim()[0]),
        #          f"Map bias: {bias_value}V",
        #          color="red", fontsize=12, alpha=0.8)

        plt.tight_layout()
        return fig, point_colors

    def save_plot(self, fig, output_path, islogy=False, dpi=500):
        """
        儲存圖表
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = "_logy" if islogy else ""
        filename = output_path.parent / f"{output_path.stem}_correlated{suffix}.png"

        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
        print(f"Correlated plot saved to: {filename}")

        return filename

    def save_plot_with_timestamp(self, fig, output_dir, base_name, islogy=False, dpi=500):
        """
        使用時間戳儲存圖表
        """
        from datetime import datetime

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "_logy" if islogy else ""
        filename = output_dir / f"{base_name}_{timestamp}_correlated{suffix}.png"

        fig.savefig(filename, dpi=dpi, bbox_inches="tight")
        print(f"Plot saved to: {filename}")

        return filename
