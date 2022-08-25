from pathlib import Path
from unittest import TestCase

import numpy as np
from scipy.io import wavfile

from entrainment import calculate_sample_correlation, calculate_time_series
from frame import Frame, MissingFrame
from interpausal_unit import InterPausalUnit
from tama import get_frames, get_interpausal_units


class TAMATestCase(TestCase):
    maxDiff = None

    def setUp(self):
        self.cases = {
            'empty': {
                'words_fname': "./data/empty.words",
                'audio_fname': "./data/empty.wav",
                'audio': wavfile.read("./data/empty.wav"),
                'expected_ipus': [],
                'expected_frames': [],
                'F0_MAX_time_series': [],
            },
            'silence': {
                'words_fname': "./data/silence.words",
                'audio_fname': "./data/silence.wav",
                'audio': wavfile.read("./data/silence.wav"),
                'expected_ipus': [],
                'expected_frames': [MissingFrame(0.0, 0.4)],
                'F0_MAX_time_series': [np.nan],
            },
            'small': {
                'words_fname': "./data/200-300-100.words",
                'audio_fname': "./data/200-300-100.wav",
                'audio': wavfile.read("./data/200-300-100.wav"),
                'expected_ipus': [
                    InterPausalUnit(0.0, 0.4),
                    InterPausalUnit(0.8, 1.6),
                    InterPausalUnit(2.0, 2.4),
                ],
                'expected_frames': [
                    Frame(
                        0.0,
                        2.4,
                        False,
                        [
                            InterPausalUnit(0.0, 0.4, {'F0_MAX': 200.002}),
                            InterPausalUnit(0.8, 1.6, {'F0_MAX': 300.002}),
                            InterPausalUnit(2.0, 2.4, {'F0_MAX': 100.003}),
                        ],
                    )
                ],
                'F0_MAX_time_series': [225.00225000000003],
            },
            'long_100-200-300': {
                'words_fname': "./data/100-200-300_long.words",
                'audio_fname': "./data/100-200-300_long.wav",
                'audio': wavfile.read("./data/100-200-300_long.wav"),
                'expected_ipus': [
                    InterPausalUnit(0.0, 4.0),
                    InterPausalUnit(8.0, 12.0),
                    InterPausalUnit(16.0, 24.0),
                ],
                'expected_frames': [
                    Frame(
                        0.0,
                        16.0,
                        False,
                        [
                            InterPausalUnit(0.0, 4.0, {'F0_MAX': 100.003}),
                            InterPausalUnit(8.0, 12.0, {'F0_MAX': 200.002}),
                        ],
                    ),
                    Frame(
                        8.0,
                        24.0,
                        False,
                        [
                            InterPausalUnit(8.0, 12.0, {'F0_MAX': 200.002}),
                            InterPausalUnit(16.0, 24.0, {'F0_MAX': 300.002}),
                        ],
                    ),
                    Frame(
                        16.0,
                        24.0,
                        False,
                        [
                            InterPausalUnit(16.0, 24.0, {'F0_MAX': 300.002}),
                        ],
                    ),
                ],
                'F0_MAX_time_series': [150.0025, 266.6686666666667, 300.002],
            },
        }

    def test_interpausal_units_separation_empty(self):
        self.assertEqual(
            get_interpausal_units(self.cases['empty']['words_fname']),
            self.cases['empty']['expected_ipus'],
        )

    def test_interpausal_units_separation_silence(self):
        self.assertEqual(
            get_interpausal_units(self.cases['silence']['words_fname']),
            self.cases['silence']['expected_ipus'],
        )

    def test_interpausal_units_separation_small(self):
        self.assertEqual(
            get_interpausal_units(self.cases['small']['words_fname']),
            self.cases['small']['expected_ipus'],
        )

    def test_interpausal_units_separation_long(self):
        self.assertEqual(
            get_interpausal_units(self.cases['long_100-200-300']['words_fname']),
            self.cases['long_100-200-300']['expected_ipus'],
        )

    def test_frame_separation_empty(self):
        case = self.cases['empty']
        self.assertEqual(
            case['expected_frames'],
            get_frames(wav_fname=case['audio_fname'], words_fname=case['words_fname']),
        )

    def test_frame_separation_silence(self):
        case = self.cases['silence']
        self.assertEqual(
            case['expected_frames'],
            get_frames(wav_fname=case['audio_fname'], words_fname=case['words_fname']),
        )

    def test_frame_separation_small(self):
        case = self.cases['small']
        self.assertEqual(
            case['expected_frames'],
            get_frames(wav_fname=case['audio_fname'], words_fname=case['words_fname']),
        )

    def test_frame_separation_long(self):
        case = self.cases['long_100-200-300']
        self.assertEqual(
            case['expected_frames'],
            get_frames(wav_fname=case['audio_fname'], words_fname=case['words_fname']),
        )

    def test_calculate_time_series_empty(self):
        case = self.cases['empty']
        self.assertEqual(
            case['F0_MAX_time_series'],
            calculate_time_series(
                "F0_MAX", case['expected_frames'], case['audio_fname']
            ),
        )

    def test_calculate_time_series_silence(self):
        case = self.cases['silence']
        self.assertEqual(
            case['F0_MAX_time_series'],
            calculate_time_series(
                "F0_MAX", case['expected_frames'], case['audio_fname']
            ),
        )

    def test_calculate_time_series_small(self):
        case = self.cases['small']
        np.testing.assert_almost_equal(
            case['F0_MAX_time_series'],
            calculate_time_series(
                "F0_MAX", case['expected_frames'], Path(case['audio_fname'])
            ),
        )

    def test_calculate_time_series_long(self):
        case = self.cases['long_100-200-300']
        np.testing.assert_almost_equal(
            case['F0_MAX_time_series'],
            calculate_time_series(
                "F0_MAX", case['expected_frames'], Path(case['audio_fname'])
            ),
        )

    def test_calculate_sample_correlation_one_empty(self):
        case = self.cases['empty']
        self.assertRaises(
            ValueError,
            calculate_sample_correlation,
            case['F0_MAX_time_series'],
            [1.0, 2.0, 3.0],  # A random non-empty list
            0,
        )

    def test_calculate_sample_correlation_long_with_itself(self):
        case = self.cases['long_100-200-300']
        self.assertSequenceEqual(
            [1.0],
            calculate_sample_correlation(
                case['F0_MAX_time_series'],
                case['F0_MAX_time_series'],
                0,
            ),
        )
