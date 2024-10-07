import unittest
from src.exchange_correlation import ExchangeCorrelation
import sys

class TestExchangeCorrelation(unittest.TestCase):

    def test_lda_potential(self):
        xc = ExchangeCorrelation(functional='LDA')
        electron_density = [1.0, 2.0, 3.0]
        V_xc = xc.calculate_potential(electron_density)
        expected_potential = sum(electron_density) / len(electron_density)
        self.assertEqual(V_xc, expected_potential)

    def test_invalid_functional(self):
        xc = ExchangeCorrelation(functional='GGA')
        electron_density = [1.0, 2.0, 3.0]
        V_xc = xc.calculate_potential(electron_density)
        # 由于GGA未实现，应返回0.0
        self.assertEqual(V_xc, 0.0)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
