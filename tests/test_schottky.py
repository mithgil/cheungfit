# tests/test_schottky.py
import unittest
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent / 'schottky' / 'modules'))


from schottky_analyzer import SchottkyAnalyzer

class TestSchottkyAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = SchottkyAnalyzer(temperature=300.0)
    
    def test_fit_schottky(self):
        # 創建測試數據
        V = np.linspace(0, 0.5, 100)
        J = 1e-6 * np.exp(V / 0.025)  # η=1 的理想 Schottky
        
        result = self.analyzer.fit_schottky(
            voltage=V,
            current=J,
            area=1.0,
            v_range=(0.1, 0.25),
            material='Si'
        )
        
        self.assertGreater(result.r_squared, 0.99)
        self.assertAlmostEqual(result.ideality_factor, 1.0, delta=0.1)

if __name__ == '__main__':
    unittest.main()