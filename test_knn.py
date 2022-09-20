from unittest import TestCase

import numpy as np
from scipy.io import wavfile

from interpausal_unit import InterPausalUnit
from knn import calculate_knn_time_series


class KNNTestCase(TestCase):
    def setUp(self):
        self.cases = {
            'long_100-200-300_x2': {
                'words_fname': "./data/100-200-300_long_x2.words",
                'audio_fname': "./data/100-200-300_long_x2.wav",
                'audio': wavfile.read("./data/100-200-300_long_x2.wav"),
                'ipus': [
                    InterPausalUnit(0.0, 4.0, {'F0_MAX': 100.003}),
                    InterPausalUnit(8.0, 12.0, {'F0_MAX': 200.002}),
                    InterPausalUnit(16.0, 24.0, {'F0_MAX': 300.002}),
                    InterPausalUnit(28.0, 32.0, {'F0_MAX': 100.003}),
                    InterPausalUnit(36.0, 40.0, {'F0_MAX': 200.002}),
                    InterPausalUnit(44.0, 52.0, {'F0_MAX': 300.002}),
                ],
                'time_series': {
                    1: {
                        'F0_MAX': [
                            100.003,
                            200.002,
                            300.002,
                            100.003,
                            200.002,
                            300.002,
                        ],
                    },
                    2: {
                        'F0_MAX': [
                            150.0025,
                            150.0025,
                            200.0025,
                            150.0025,
                            150.0025,
                            250.002,
                        ],
                    },
                    3: {
                        'F0_MAX': [
                            200.002333333,
                            200.002333333,
                            200.002333333,
                            200.002333333,
                            200.002333333,
                            200.002333333,
                        ],
                    },
                },
            },
        }

    def test_calculate_knn_time_series_longx2_k_1(self):
        case = self.cases['long_100-200-300_x2']
        np.testing.assert_almost_equal(
            case['time_series'][1]['F0_MAX'],
            calculate_knn_time_series(
                1, 'F0_MAX', case['ipus'], case['audio_fname'], 'praat'
            ),
        )

    def test_calculate_knn_time_series_longx2_k_2(self):
        case = self.cases['long_100-200-300_x2']
        np.testing.assert_almost_equal(
            case['time_series'][2]['F0_MAX'],
            calculate_knn_time_series(
                2, 'F0_MAX', case['ipus'], case['audio_fname'], 'praat'
            ),
        )

    def test_calculate_knn_time_series_longx2_k_3(self):
        case = self.cases['long_100-200-300_x2']
        np.testing.assert_almost_equal(
            case['time_series'][3]['F0_MAX'],
            calculate_knn_time_series(
                3, 'F0_MAX', case['ipus'], case['audio_fname'], 'praat'
            ),
        )
