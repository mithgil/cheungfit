# src/ivs_tools/physics/schottky_analyzer.py 
from .sliding_window_fitter import SlidingWindowCheungFitter, FittingConfig, FittingResult

class SchottkyAnalyzer:
    """原有功能 + 新滑動窗口擬合"""
    
    def __init__(self, area_cm2: float, T: float = 300, a_star: float = 54.0):
        self.area_cm2 = area_cm2
        self.T = T
        self.a_star = a_star
        self.fitter = SlidingWindowCheungFitter()
        
    def fit_pixel_sliding(self, bias: np.ndarray, current: np.ndarray, 
                         layer_type: str = "multilayer") -> FittingResult:
        """對單一像素執行滑動窗口擬合"""
        return self.fitter.fit(
            bias=bias, 
            current=current, 
            area_cm2=self.area_cm2, 
            T=self.T, 
            a_star=self.a_star,
            layer_type=layer_type
        )
    
    def batch_fit_map(self, h5map, layer_type_map: Optional[np.ndarray] = None,
                     pixel_indices: Optional[List[Tuple[int,int]]] = None) -> Dict:
        """
        批量擬合整個 mapping
        
        Parameters:
        -----------
        h5map : HDF5Analyzer
            已載入的 HDF5 分析器
        layer_type_map : np.ndarray, optional
            與 spatial map 同形狀的陣列，標註每個像素的層數 ("mono"/"multi")
        pixel_indices : list of tuple, optional
            若指定，僅處理這些像素 (用於人工監督模式)
        """
        from tqdm import tqdm  # 進度條
        
        metadata = h5map.metadata
        total_pixels = metadata.x_pixels * metadata.y_pixels
        
        if pixel_indices:
            pixels_to_process = pixel_indices
        else:
            pixels_to_process = [(y, x) for y in range(metadata.y_pixels) 
                                for x in range(metadata.x_pixels)]
        
        results = {}
        
        for y, x in tqdm(pixels_to_process, desc="Fitting pixels"):
            # 獲取該像素的 I-V
            spectrum = h5map.get_spectrum_at_pixel(x, y)[3:]  # 跳过前幾個點（可調整）
            bias = h5map.bias_vector[3:]
            
            # 確定層數
            layer = "multilayer"
            if layer_type_map is not None:
                layer = layer_type_map[y, x]
            
            # 執行擬合
            result = self.fit_pixel_sliding(bias, spectrum, layer_type=layer)
            
            if result.success:
                results[(x, y)] = {
                    'n': result.n,
                    'Rs': result.Rs,
                    'phi_b': result.phi_b,
                    'r2': result.r_squared,
                    'fit_range': result.fit_range,
                    'score': result.score
                }
            else:
                results[(x, y)] = {
                    'success': False,
                    'warning': result.warning,
                    'fit_range': None
                }
        
        return results