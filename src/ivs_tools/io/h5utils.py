# h5utils.py
# 2026/1/19

import h5py
import numpy as np
import json
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path

@dataclass
class HDF5DatasetInfo:
    """儲存 HDF5 資料集的資訊"""
    name: str
    path: str
    shape: tuple
    dtype: str
    data: Optional[np.ndarray] = None
    attributes: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.attributes is None:
            self.attributes = {}

@dataclass
class HDF5Metadata:
    """專門處理 metadata 的類別"""
    bias_points: int
    x_pixels: int
    y_pixels: int
    x_range: float
    y_range: float
    experiment_type: str
    unit: str
    raw_metadata: Dict[str, Any] = None
    
    @classmethod
    def from_hdf5_group(cls, group: h5py.Group) -> 'HDF5Metadata':
        """從 HDF5 group 創建 Metadata 物件"""
        attrs = dict(group.attrs)
        
        # 解析 raw_metadata
        raw_metadata = {}
        if 'raw_metadata' in group:
            raw_metadata_group = group['raw_metadata']
            if 'metadata_json' in raw_metadata_group.attrs:
                try:
                    raw_metadata = json.loads(raw_metadata_group.attrs['metadata_json'])
                except:
                    raw_metadata = {'raw_json': raw_metadata_group.attrs['metadata_json']}
        
        return cls(
            bias_points=int(attrs.get('bias_points', 0)),
            x_pixels=int(attrs.get('x_pixels', 0)),
            y_pixels=int(attrs.get('y_pixels', 0)),
            x_range=float(attrs.get('x_range', 0.0)),
            y_range=float(attrs.get('y_range', 0.0)),
            experiment_type=attrs.get('experiment_type', 'Unknown'),
            unit=attrs.get('unit', 'Unknown'),
            raw_metadata=raw_metadata
        )

class HDF5Analyzer:
    def __init__(self, filepath: Union[str, Path], auto_load: bool = True):
        """
        初始化 HDF5 分析器
        
        Args:
            filepath: HDF5 檔案路徑
            auto_load: 是否自動載入資料
        """         

        self.filepath = Path(filepath)
        # 驗證檔案        
        self._validate_file()
        self._datasets: Dict[str, HDF5DatasetInfo] = {}
        self._metadata: Optional[HDF5Metadata] = None
        self._current_map: Optional[np.ndarray] = None
        self._bias_vector: Optional[np.ndarray] = None
        
        if auto_load:
            self.load()
    
    def __repr__(self) -> str:
        return f"HDF5Analyzer(file='{self.filepath.name}', datasets={list(self._datasets.keys())})"

    def _validate_file(self):
        """驗證檔案是否存在且是 HDF5 格式"""
        if not self.filepath.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.filepath}")
        
        if not self.filepath.is_file():
            raise ValueError(f"Path is not a file: {self.filepath}")
        
        # 檢查副檔名（可選）
        if self.filepath.suffix.lower() not in ['.h5', '.hdf5', '.hdf']:
            print(f"Warning: File extension '{self.filepath.suffix}' might not be HDF5 format")
            
    def load(self) -> None:
        """載入 HDF5 檔案的所有資料"""
        if not self.filepath.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.filepath}")
        
        try:
            with h5py.File(self.filepath, 'r') as f:
                self._parse_datasets(f)
                self._parse_metadata(f)
                self._cache_important_data()
        except Exception as e:
            raise IOError(f"Failed to load HDF5 file: {e}")
    
    def _parse_datasets(self, h5file: h5py.File) -> None:
        """解析所有的 dataset"""
        self._datasets.clear()
        
        def _process_dataset(name: str, obj: h5py.Dataset):
            if isinstance(obj, h5py.Dataset):
                dataset_info = HDF5DatasetInfo(
                    name=Path(name).name,
                    path=name,
                    shape=obj.shape,
                    dtype=str(obj.dtype),
                    attributes=dict(obj.attrs)
                )
                self._datasets[name] = dataset_info
        
        h5file.visititems(_process_dataset)
    
    def _parse_metadata(self, h5file: h5py.File) -> None:
        """解析 metadata"""
        if 'metadata' in h5file:
            self._metadata = HDF5Metadata.from_hdf5_group(h5file['metadata'])
    
    def _cache_important_data(self) -> None:
        """快取重要的數據到記憶體"""
        if 'bias' in self._datasets:
            with h5py.File(self.filepath, 'r') as f:
                self._bias_vector = f['bias'][:]
        
        if 'current_fwd' in self._datasets:
            with h5py.File(self.filepath, 'r') as f:
                self._current_map = f['current_fwd'][:]
    
    @property
    def bias_vector(self) -> np.ndarray:
        """取得 bias 向量"""
        if self._bias_vector is None:
            self.load_data('bias')
        return self._bias_vector
    
    @property
    def current_map(self) -> np.ndarray:
        """取得 current map"""
        if self._current_map is None:
            self.load_data('current_fwd')
        return self._current_map
    
    @property
    def metadata(self) -> HDF5Metadata:
        """取得 metadata"""
        if self._metadata is None:
            self.load()
        return self._metadata
    
    def load_data(self, dataset_name: str, lazy: bool = False) -> Optional[np.ndarray]:
        """
        載入特定 dataset 的數據
        
        Args:
            dataset_name: dataset 名稱或路徑
            lazy: 如果為 True，只返回參考而不載入數據
            
        Returns:
            dataset 數據或 None
        """
        if dataset_name not in self._datasets:
            # 嘗試用路徑查找
            for path, info in self._datasets.items():
                if info.name == dataset_name or path.endswith(dataset_name):
                    dataset_name = path
                    break
            else:
                raise KeyError(f"Dataset '{dataset_name}' not found")
        
        info = self._datasets[dataset_name]
        
        if lazy:
            return None
        
        if info.data is None:
            with h5py.File(self.filepath, 'r') as f:
                info.data = f[info.path][:]
        
        # 更新快取
        if info.path == '/bias':
            self._bias_vector = info.data
        elif info.path == '/current_fwd':
            self._current_map = info.data
        
        return info.data
    
    def list_structure(self, verbose: bool = True) -> Dict[str, Any]:
        """
        列出 HDF5 結構
        
        Args:
            verbose: 是否打印詳細資訊
            
        Returns:
            結構字典
        """
        structure = {}
        
        for path, info in self._datasets.items():
            if verbose:
                print(f"DATASET: {path} (Shape: {info.shape}, Dtype: {info.dtype})")
                if info.attributes:
                    for key, value in info.attributes.items():
                        print(f"  - ATTR: {key} = {value}")
            
            structure[path] = {
                'shape': info.shape,
                'dtype': info.dtype,
                'attributes': info.attributes
            }
        
        if self._metadata and verbose:
            print("\nMETADATA:")
            print(f"  Experiment: {self._metadata.experiment_type}")
            print(f"  Bias Points: {self._metadata.bias_points}")
            print(f"  Pixels: {self._metadata.x_pixels} x {self._metadata.y_pixels}")
            print(f"  Range: {self._metadata.x_range:.2e} x {self._metadata.y_range:.2e} {self._metadata.unit}")
        
        return structure
    
    def get_nearest_bias_index(self, bias_value: float) -> int:
        """
        取得最接近指定 bias 值的索引
        
        Args:
            bias_value: 目標 bias 值
            
        Returns:
            最接近的索引
        """
        bias_vec = self.bias_vector
        index = int(np.argmin(np.abs(bias_vec - bias_value)))
        return index
    
    def get_current_at_bias(self, bias_value: float) -> np.ndarray:
        """
        取得指定 bias 下的 current map
        
        Args:
            bias_value: bias 值
            
        Returns:
            指定 bias 下的 current map (shape: (x_pixels * y_pixels,))
        """
        idx = self.get_nearest_bias_index(bias_value)
        current_map = self.current_map
        return current_map[idx, :]
    
    def get_spectrum_at_pixel(self, x: int, y: int) -> np.ndarray:
        """
        取得指定像素點的 spectrum
        
        Args:
            x: X 座標
            y: Y 座標
            
        Returns:
            該像素點的 spectrum (shape: (bias_points,))
        """
        if not self._metadata:
            raise ValueError("Metadata not loaded")
        
        if x >= self._metadata.x_pixels or y >= self._metadata.y_pixels:
            raise ValueError(f"Pixel coordinates out of range. Max: ({self._metadata.x_pixels-1}, {self._metadata.y_pixels-1})")
        
        pixel_index = x + y * self._metadata.x_pixels
        current_map = self.current_map
        return current_map[:, pixel_index]
    
    def get_pixel_coordinates(self, pixel_index: int) -> tuple:
        """
        從像素索引取得座標
        
        Args:
            pixel_index: 像素索引
            
        Returns:
            (x, y) 座標
        """
        if not self._metadata:
            raise ValueError("Metadata not loaded")
        
        x = pixel_index % self._metadata.x_pixels
        y = pixel_index // self._metadata.x_pixels
        return x, y
    
    def reshape_to_2d(self, data_1d: np.ndarray) -> np.ndarray:
        """
        將 1D 數據重塑為 2D 網格
        
        Args:
            data_1d: 1D 數據 (長度應為 x_pixels * y_pixels)
            
        Returns:
            2D 網格數據
        """
        if not self._metadata:
            raise ValueError("Metadata not loaded")
        
        expected_size = self._metadata.x_pixels * self._metadata.y_pixels
        if len(data_1d) != expected_size:
            raise ValueError(f"Data size mismatch. Expected {expected_size}, got {len(data_1d)}")
        
        return data_1d.reshape((self._metadata.y_pixels, self._metadata.x_pixels))
    
    def get_2d_map_at_bias(self, bias_value: float) -> np.ndarray:
        """
        取得指定 bias 下的 2D current map
        
        Args:
            bias_value: bias 值
            
        Returns:
            2D current map (shape: (y_pixels, x_pixels))
        """
        current_1d = self.get_current_at_bias(bias_value)
        return self.reshape_to_2d(current_1d)
        
    def print_usage(self) -> None:
        """
        打印使用指南
        
        顯示類的主要屬性和方法，以及如何使用它們
        """
        print("=" * 60)
        print("HDF5Analyzer 使用指南")
        print("=" * 60)
        print()
        
        # 顯示基本屬性
        print("📁 數據屬性:")
        print("-" * 40)
        for prop_name in ['filepath', 'metadata', 'current_map', 'bias_vector',
                         'bias_points', 'x_pixels', 'y_pixels']:
            if hasattr(self, prop_name):
                try:
                    value = getattr(self, prop_name)
                    desc = self._property_descriptions.get(prop_name, '')
                    
                    if prop_name in ['current_map', 'bias_vector']:
                        # 對於數組，顯示形狀
                        if value is not None:
                            print(f"  • {prop_name}: {desc}")
                            print(f"    形狀: {value.shape}, 類型: {value.dtype}")
                        else:
                            print(f"  • {prop_name}: {desc} (未加載)")
                    elif prop_name == 'metadata':
                        if value is not None:
                            print(f"  • {prop_name}: {desc}")
                            print(f"    文件: {value.filename}")
                            print(f"    尺寸: {value.x_pixels}×{value.y_pixels}, 偏壓點: {value.bias_points}")
                        else:
                            print(f"  • {prop_name}: {desc} (未加載)")
                    else:
                        print(f"  • {prop_name}: {value} - {desc}")
                except (ValueError, AttributeError):
                    print(f"  • {prop_name}: 未加載或不可用")
        
        print()
        print("⚙️ 主要方法:")
        print("-" * 40)
        methods = [
            ("load_metadata()", "加載元數據"),
            ("load_current_map()", "加載電流映射"),
            ("load_bias_vector()", "加載偏壓序列"),
            ("load_all_data()", "加載所有數據"),
            ("get_data_info()", "獲取數據信息"),
            ("print_usage()", "顯示使用指南"),
            ("create_slice_viewer()", "創建切片查看器 (需要額外導入)")
        ]
        
        for method, desc in methods:
            print(f"  • {method}: {desc}")
        
        print()
        print("🔧 數據操作示例:")
        print("-" * 40)
        examples = [
            "# 創建分析器並加載數據",
            "analyzer = HDF5Analyzer('data.h5')",
            "",
            "# 訪問數據屬性",
            "current_map = analyzer.current_map  # 3D電流數據",
            "bias_voltage = analyzer.bias_vector  # 偏壓序列",
            "pixels_x = analyzer.metadata.x_pixels  # X像素數",
            "",
            "# 獲取特定偏壓的切片",
            "bias_index = 250",
            "current_slice = analyzer.current_map[bias_index, :]",
            "",
            "# 重塑為2D圖像",
            "current_2d = current_slice.reshape(analyzer.metadata.y_pixels, analyzer.metadata.x_pixels)",
            "",
            "# 獲取特定像素的I-V曲線",
            "pixel_x, pixel_y = 50, 50",
            "pixel_index = pixel_y * analyzer.metadata.x_pixels + pixel_x",
            "iv_curve = analyzer.current_map[:, pixel_index]",
        ]
        
        for example in examples:
            print(f"  {example}")
        
        print()
        print("📊 可視化示例:")
        print("-" * 40)
        print("  # 創建交互式切片查看器")
        print("  from visualization import create_slice_viewer")
        print("  create_slice_viewer(analyzer)")
        print()
        print("=" * 60)
    
    def print_quick_guide(self) -> None:
        """快速使用指南"""
        print("🚀 HDF5Analyzer 快速指南")
        print("=" * 40)
        print("1. 創建分析器:")
        print("   analyzer = HDF5Analyzer('your_file.h5')")
        print()
        print("2. 訪問主要數據:")
        print("   analyzer.current_map    # 3D電流數據")
        print("   analyzer.bias_vector    # 偏壓序列")
        print("   analyzer.metadata       # 實驗參數")
        print()
        print("3. 獲取尺寸信息:")
        print("   analyzer.bias_points    # 偏壓點數")
        print("   analyzer.x_pixels       # X像素數")
        print("   analyzer.y_pixels       # Y像素數")
        print()
        print("4. 完整指南:")
        print("   analyzer.print_usage()  # 顯示詳細指南")
        print("=" * 40)
    
    def __str__(self) -> str:
        """字符串表示"""
        info = self.get_data_info()
        
        lines = [
            f"HDF5Analyzer: {info['filename']}",
            f"  Current Map: {info['current_map_shape']}" if info['current_map_shape'] else "  Current Map: 未加載",
            f"  Bias Vector: {info['bias_vector_shape']}" if info['bias_vector_shape'] else "  Bias Vector: 未加載",
            f"  空間分辨率: {info['x_pixels']}×{info['y_pixels']} 像素",
            f"  偏壓點數: {info['bias_points']}",
        ]
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """代表字符串"""
        return f"HDF5Analyzer(filepath='{self.filepath}')"
    
    def summary(self) -> None:
        """顯示數據摘要"""
        print("📋 數據摘要")
        print("-" * 40)
        print(f"文件: {self.filepath.name}")
        print(f"路徑: {self.filepath}")
        print()
        
        if self._metadata is not None:
            print("📐 空間維度:")
            print(f"  X方向像素: {self.metadata.x_pixels}")
            print(f"  Y方向像素: {self.metadata.y_pixels}")
            print()
            
            print("⚡ 偏壓維度:")
            print(f"  偏壓點數: {self.metadata.bias_points}")
            if self._bias_vector is not None:
                print(f"  偏壓範圍: {self.bias_vector.min():.3f} 到 {self.bias_vector.max():.3f} V")
            print()
            
            print("💾 數據陣列:")
            if self._current_map is not None:
                print(f"  Current Map: {self.current_map.shape} ({self.current_map.dtype})")
                print(f"    總大小: {self.current_map.nbytes / 1024**2:.2f} MB")
                print(f"    範圍: {self.current_map.min():.3e} 到 {self.current_map.max():.3e} A")
            
            if self._bias_vector is not None:
                print(f"  Bias Vector: {self.bias_vector.shape} ({self.bias_vector.dtype})")
        else:
            print("⚠️ 數據尚未加載")
            print("使用 analyzer.load_all_data() 加載數據")
        
        print("-" * 40)
    