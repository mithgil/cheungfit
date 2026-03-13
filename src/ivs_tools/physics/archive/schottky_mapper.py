# schottky_mapper_optimized.py
# 2026/1/19

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import ndimage
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SchottkyMapper:
    def __init__(self, h5_analyzer, phys_analyzer, area_cm2, 
                 fit_range=(0.1, 0.25), sg_window=15,
                 min_r2=0.95, max_n=10.0, skip_points=3):
        """
        Initialize Schottky Mapper
        
        Args:
            h5_analyzer: HDF5Analyzer instance
            phys_analyzer: SchottkyAnalyzer instance
            area_cm2: contact area (cm^2)
            fit_range: voltage range for fitting (V)
            sg_window: Savitzky-Golay filter window
            min_r2: minimum R^2 for valid fit
            max_n: maximum ideality factor for valid fit
            skip_points: number of initial points to skip
        """
        self.h5 = h5_analyzer
        self.phys = phys_analyzer
        self.area = area_cm2
        self.fit_range = fit_range
        self.sg_window = sg_window
        self.min_r2 = min_r2
        self.max_n = max_n
        self.skip = skip_points
        
        # Results arrays
        self.n_map = None
        self.phi_map = None
        self.j0_map = None
        self.r2_map = None
        self.slope_map = None
        self.valid_mask = None
        self.stats = None
        
        # Preprocess data
        self.bias = self.h5.bias_vector[skip_points:]
        self.current = self.h5.current_map[skip_points:, :]
        
        self.x_pixels = self.h5.metadata.x_pixels
        self.y_pixels = self.h5.metadata.y_pixels
        self.total_pixels = self.x_pixels * self.y_pixels
        
        print(f"SchottkyMapper initialized")
        print(f"  Bias shape: {self.bias.shape}")
        print(f"  Current shape: {self.current.shape}")
        print(f"  Fit range: {fit_range}")
        print(f"  Area: {area_cm2:.2e} cm^2")
    
    def fit_all_pixels(self, show_progress=True):
        """
        Fit all pixels in the map
        
        Args:
            show_progress: show progress bar
        """
        # Initialize result arrays
        self.n_map = np.full((self.y_pixels, self.x_pixels), np.nan)
        self.phi_map = np.full((self.y_pixels, self.x_pixels), np.nan)
        self.j0_map = np.full((self.y_pixels, self.x_pixels), np.nan)
        self.r2_map = np.full((self.y_pixels, self.x_pixels), np.nan)
        self.slope_map = np.full((self.y_pixels, self.x_pixels), np.nan)
        
        valid_count = 0
        
        # Setup progress bar
        if show_progress:
            pbar = tqdm(total=self.total_pixels, 
                       desc="Fitting pixels",
                       unit="pix",
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        # Process each pixel
        for idx in range(self.total_pixels):
            y, x = divmod(idx, self.x_pixels)
            spectrum = self.current[:, idx]
            
            result = self.phys.process_pixel(
                self.bias, spectrum, self.area,
                fit_range=self.fit_range, 
                sg_window=self.sg_window
            )
            
            if result and result['r2'] >= self.min_r2 and result['n'] <= self.max_n:
                self.n_map[y, x] = result['n']
                self.phi_map[y, x] = result['phi_b']
                self.j0_map[y, x] = result['j0']
                self.r2_map[y, x] = result['r2']
                self.slope_map[y, x] = result['slope']
                valid_count += 1
            
            if show_progress:
                pbar.update(1)
                if idx % 100 == 0:
                    pbar.set_postfix({'valid': f'{valid_count}/{idx+1}'})
        
        if show_progress:
            pbar.close()
        
        # Create valid mask
        self.valid_mask = ~np.isnan(self.n_map)
        
        # Calculate statistics
        self.stats = {
            'total': self.total_pixels,
            'valid': valid_count,
            'ratio': valid_count / self.total_pixels * 100,
            'n_mean': np.nanmean(self.n_map),
            'n_std': np.nanstd(self.n_map),
            'phi_mean': np.nanmean(self.phi_map),
            'phi_std': np.nanstd(self.phi_map),
            'j0_mean': np.nanmean(self.j0_map),
            'j0_std': np.nanstd(self.j0_map),
        }
        
        print(f"Fit complete: {valid_count}/{self.total_pixels} valid ({self.stats['ratio']:.1f}%)")
        print(f"Ideality factor: {self.stats['n_mean']:.2f} ± {self.stats['n_std']:.2f}")
        print(f"Barrier height: {self.stats['phi_mean']:.3f} ± {self.stats['phi_std']:.3f} eV")
    
    def filter_maps(self, size=3):
        """
        Apply median filter to remove outliers
        
        Args:
            size: filter kernel size
        """
        if self.n_map is None:
            raise ValueError("Run fit_all_pixels() first")
        
        maps = ['n_map', 'phi_map', 'j0_map', 'slope_map']
        
        for map_name in maps:
            data = getattr(self, map_name)
            filtered = ndimage.median_filter(data, size=size)
            data[self.valid_mask] = filtered[self.valid_mask]
            setattr(self, map_name, data)
        
        print(f"Applied {size}x{size} median filter")
    
    def plot_maps(self, figsize=(15, 10)):
        """
        Plot all parameter maps
        """
        if self.n_map is None:
            raise ValueError("Run fit_all_pixels() first")
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes = axes.flatten()
        
        # 1. Ideality factor
        im1 = axes[0].imshow(self.n_map, cmap='RdYlBu_r', vmin=1, vmax=3, origin='lower')
        axes[0].set_title('Ideality Factor (n)')
        plt.colorbar(im1, ax=axes[0], shrink = 0.4, aspect = 7)
        
        # 2. Barrier height
        im2 = axes[1].imshow(self.phi_map, cmap='viridis', vmin=0.1, vmax=0.8, origin='lower')
        axes[1].set_title('Barrier Height (eV)')
        plt.colorbar(im2, ax=axes[1], shrink = 0.4, aspect = 7)
        
        # 3. Saturation current
        j0_valid = self.j0_map[self.valid_mask]
        if len(j0_valid) > 0:
            vmin, vmax = np.percentile(j0_valid, [5, 95])
            im3 = axes[2].imshow(self.j0_map, cmap='plasma', norm=LogNorm(vmin=vmin, vmax=vmax), origin='lower')
            axes[2].set_title('J₀ (A/cm²)')
            plt.colorbar(im3, ax=axes[2], shrink = 0.4, aspect = 7)
        
        # 4. R²
        im4 = axes[3].imshow(self.r2_map, cmap='Greens', vmin=0.9, vmax=1.0, origin='lower')
        axes[3].set_title('R² of Fit')
        plt.colorbar(im4, ax=axes[3], shrink = 0.4, aspect = 7)
        
        # 5. Slope
        im5 = axes[4].imshow(self.slope_map, cmap='coolwarm', origin='lower')
        axes[4].set_title('Slope of ln(J) vs V')
        plt.colorbar(im5, ax=axes[4], shrink = 0.4, aspect = 7)
        
        # 6. Valid mask
        im6 = axes[5].imshow(self.valid_mask.astype(float), cmap='binary', origin='lower')
        axes[5].set_title(f'Valid Pixels ({self.stats["ratio"]:.1f}%)')
        plt.colorbar(im6, ax=axes[5], shrink = 0.4, aspect = 7)
        
        fig.suptitle(f'Schottky Analysis - {self.h5.filepath.name}', fontsize=14)
        plt.tight_layout()
        return fig
    
    def plot_histograms(self, figsize=(12, 8)):
        """
        Plot parameter distributions
        """
        if self.n_map is None:
            raise ValueError("Run fit_all_pixels() first")
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Get valid data
        n_valid = self.n_map[self.valid_mask]
        phi_valid = self.phi_map[self.valid_mask]
        j0_valid = self.j0_map[self.valid_mask]
        r2_valid = self.r2_map[self.valid_mask]
        
        # 1. Ideality factor
        axes[0, 0].hist(n_valid, bins=50, density=True, alpha=0.7, color='skyblue')
        axes[0, 0].axvline(np.mean(n_valid), color='red', linestyle='--')
        axes[0, 0].set_xlabel('n')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Ideality Factor Distribution')
        
        # 2. Barrier height
        axes[0, 1].hist(phi_valid, bins=50, density=True, alpha=0.7, color='lightgreen')
        axes[0, 1].axvline(np.mean(phi_valid), color='red', linestyle='--')
        axes[0, 1].set_xlabel('Φ_B (eV)')
        axes[0, 1].set_title('Barrier Height Distribution')
        
        # 3. Saturation current
        if len(j0_valid) > 0:
            axes[1, 0].hist(np.log10(j0_valid), bins=50, density=True, alpha=0.7, color='salmon')
            axes[1, 0].axvline(np.mean(np.log10(j0_valid)), color='red', linestyle='--')
            axes[1, 0].set_xlabel('log₁₀(J₀)')
            axes[1, 0].set_title('J₀ Distribution')
        
        # 4. R²
        axes[1, 1].hist(r2_valid, bins=50, density=True, alpha=0.7, color='gold')
        axes[1, 1].axvline(np.mean(r2_valid), color='red', linestyle='--')
        axes[1, 1].axvline(self.min_r2, color='green', linestyle=':')
        axes[1, 1].set_xlabel('R²')
        axes[1, 1].set_title('R² Distribution')
        
        plt.suptitle('Parameter Distributions')
        plt.tight_layout()
        return fig
    
    def get_pixel(self, x, y):
        """
        Get results for specific pixel
        
        Args:
            x, y: pixel coordinates
        """
        if not self.valid_mask[y, x]:
            return None
        
        return {
            'x': x, 'y': y,
            'n': self.n_map[y, x],
            'phi_b': self.phi_map[y, x],
            'j0': self.j0_map[y, x],
            'r2': self.r2_map[y, x],
            'slope': self.slope_map[y, x]
        }
    
    def save_results(self, output_dir='schottky_results'):
        """
        Save all results to files
        """
        if self.n_map is None:
            raise ValueError("Run fit_all_pixels() first")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        base_name = self.h5.filepath.stem
        
        # Save as NPZ
        npz_file = output_path / f"{base_name}_results.npz"
        np.savez_compressed(
            npz_file,
            n=self.n_map,
            phi=self.phi_map,
            j0=self.j0_map,
            r2=self.r2_map,
            slope=self.slope_map,
            mask=self.valid_mask,
            bias=self.bias,
            stats=self.stats
        )
        
        # Save plots
        fig1 = self.plot_maps()
        fig1.savefig(output_path / f"{base_name}_maps.png", dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        fig2 = self.plot_histograms()
        fig2.savefig(output_path / f"{base_name}_histograms.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        # Save text report
        report_file = output_path / f"{base_name}_report.txt"
        with open(report_file, 'w') as f:
            f.write("Schottky Analysis Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"File: {self.h5.filepath.name}\n")
            f.write(f"Valid pixels: {self.stats['valid']}/{self.stats['total']} ({self.stats['ratio']:.1f}%)\n\n")
            f.write(f"Ideality factor: {self.stats['n_mean']:.3f} ± {self.stats['n_std']:.3f}\n")
            f.write(f"Barrier height: {self.stats['phi_mean']:.3f} ± {self.stats['phi_std']:.3f} eV\n")
            f.write(f"Saturation current: {self.stats['j0_mean']:.2e} ± {self.stats['j0_std']:.2e} A/cm²\n")
        
        print(f"Results saved to: {output_path}")
        print(f"  - {npz_file}")
        print(f"  - {output_path / f'{base_name}_maps.png'}")
        print(f"  - {output_path / f'{base_name}_histograms.png'}")
        print(f"  - {report_file}")


# ============ 使用示例 ============

def main_example():
    """
    Example usage of SchottkyMapper
    """
    # Create analyzers
    h5_analyzer = HDF5Analyzer("your_data.h5")
    phys_analyzer = SchottkyAnalyzer(temperature=300, a_star=54.0)
    
    # Calculate area
    d_nm = 25
    area_cm2 = np.pi * (d_nm/2 * 1e-7)**2
    
    # Create mapper
    mapper = SchottkyMapper(
        h5_analyzer=h5_analyzer,
        phys_analyzer=phys_analyzer,
        area_cm2=area_cm2,
        fit_range=(0.1, 0.25),
        sg_window=15,
        min_r2=0.95,
        max_n=10.0
    )
    
    # Fit all pixels (with progress bar)
    mapper.fit_all_pixels(show_progress=True)
    
    # Apply filter
    mapper.filter_maps(size=3)
    
    # Plot results
    mapper.plot_maps()
    mapper.plot_histograms()
    
    # Save results
    mapper.save_results()
    
    # Get specific pixel
    result = mapper.get_pixel(42, 55)
    if result:
        print(f"\nPixel (42, 55):")
        print(f"  n = {result['n']:.2f}")
        print(f"  Φ_B = {result['phi_b']:.3f} eV")
        print(f"  J₀ = {result['j0']:.2e} A/cm²")
    
    plt.show()


if __name__ == "__main__":
    # Uncomment to run example
    # main_example()
    pass