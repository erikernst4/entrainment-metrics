import warnings
from unittest import TestCase

import numpy as np
from scipy.io import wavfile

from speechalyze import InterPausalUnit, tama
from speechalyze.utils import get_interpausal_units


class TAMATestCase(TestCase):
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
                'expected_frames': [tama.MissingFrame(0.0, 0.4)],
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
                    tama.Frame(
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
                    tama.Frame(
                        0.0,
                        16.0,
                        False,
                        [
                            InterPausalUnit(0.0, 4.0, {'F0_MAX': 100.003}),
                            InterPausalUnit(8.0, 12.0, {'F0_MAX': 200.002}),
                        ],
                    ),
                    tama.Frame(
                        8.0,
                        24.0,
                        False,
                        [
                            InterPausalUnit(8.0, 12.0, {'F0_MAX': 200.002}),
                            InterPausalUnit(16.0, 24.0, {'F0_MAX': 300.002}),
                        ],
                    ),
                    tama.Frame(
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
            'long_100-200-300_x2': {
                'words_fname': "./data/100-200-300_long_x2.words",
                'audio_fname': "./data/100-200-300_long_x2.wav",
                'audio': wavfile.read("./data/100-200-300_long_x2.wav"),
                'expected_ipus': [
                    InterPausalUnit(0.0, 4.0),
                    InterPausalUnit(8.0, 12.0),
                    InterPausalUnit(16.0, 24.0),
                    InterPausalUnit(28.0, 32.0),
                    InterPausalUnit(36.0, 40.0),
                    InterPausalUnit(44.0, 52.0),
                ],
                'expected_frames': [
                    tama.Frame(
                        0.0,
                        16.0,
                        False,
                        [
                            InterPausalUnit(0.0, 4.0, {'F0_MAX': 100.003}),
                            InterPausalUnit(8.0, 12.0, {'F0_MAX': 200.002}),
                        ],
                    ),
                    tama.Frame(
                        8.0,
                        24.0,
                        False,
                        [
                            InterPausalUnit(8.0, 12.0, {'F0_MAX': 200.002}),
                            InterPausalUnit(16.0, 24.0, {'F0_MAX': 300.002}),
                        ],
                    ),
                    tama.Frame(
                        16.0,
                        32.0,
                        False,
                        [
                            InterPausalUnit(16.0, 24.0, {'F0_MAX': 300.002}),
                            InterPausalUnit(28.0, 32.0, {'F0_MAX': 100.003}),
                        ],
                    ),
                    tama.Frame(
                        24.0,
                        40.0,
                        False,
                        [
                            InterPausalUnit(28.0, 32.0, {'F0_MAX': 100.003}),
                            InterPausalUnit(36.0, 40.0, {'F0_MAX': 200.002}),
                        ],
                    ),
                    tama.Frame(
                        32.0,
                        48.0,
                        False,
                        [
                            InterPausalUnit(36.0, 40.0, {'F0_MAX': 200.002}),
                            InterPausalUnit(44.0, 52.0, {'F0_MAX': 300.002}),
                        ],
                    ),
                    tama.Frame(
                        40.0,
                        52.0,
                        False,
                        [
                            InterPausalUnit(44.0, 52.0, {'F0_MAX': 300.002}),
                        ],
                    ),
                    tama.Frame(
                        48.0,
                        52.0,
                        False,
                        [
                            InterPausalUnit(44.0, 52.0, {'F0_MAX': 300.002}),
                        ],
                    ),
                ],
                'F0_MAX_time_series': [
                    150.0025,
                    266.6686666666667,
                    233.33566666666667,
                    150.0025,
                    266.6686666666667,
                    300.002,
                    300.002,
                ],
            },
            'spoken': {
                'words_fname': "./data/hola-camaron.words",
                'audio_fname': "./data/hola-camaron.wav",
                'audio': wavfile.read("./data/hola-camaron.wav"),
                'expected_ipus': [
                    InterPausalUnit(0.0, 0.342604),
                    InterPausalUnit(0.742604, 1.085208),
                    InterPausalUnit(1.485208, 2.129437),
                ],
                'expected_frames': [
                    tama.Frame(
                        0.0,
                        2.1294375,
                        False,
                        [
                            InterPausalUnit(0.0, 0.342604, {'F0_MAX': 103.970}),
                            InterPausalUnit(0.742604, 1.085208, {'F0_MAX': 103.970}),
                            InterPausalUnit(1.485208, 2.129437, {'F0_MAX': 92.121}),
                        ],
                    )
                ],
                'F0_MAX_time_series': [98.22811872168445],
                'F0final_sma_de_maxPos_time_series': [0.6340424008061056],
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

    def test_interpausal_units_separation_long_x2(self):
        self.assertEqual(
            get_interpausal_units(self.cases['long_100-200-300_x2']['words_fname']),
            self.cases['long_100-200-300_x2']['expected_ipus'],
        )

    def test_interpausal_units_separation_spoken(self):
        self.assertEqual(
            get_interpausal_units(self.cases['spoken']['words_fname']),
            self.cases['spoken']['expected_ipus'],
        )

    def test_frame_separation_empty(self):
        case = self.cases['empty']
        self.assertEqual(
            case['expected_frames'],
            tama.get_frames(
                wav_fname=case['audio_fname'], words_fname=case['words_fname']
            ),
        )

    def test_frame_separation_silence(self):
        case = self.cases['silence']
        self.assertEqual(
            case['expected_frames'],
            tama.get_frames(
                wav_fname=case['audio_fname'], words_fname=case['words_fname']
            ),
        )

    def test_frame_separation_small(self):
        case = self.cases['small']
        self.assertEqual(
            case['expected_frames'],
            tama.get_frames(
                wav_fname=case['audio_fname'], words_fname=case['words_fname']
            ),
        )

    def test_frame_separation_long(self):
        case = self.cases['long_100-200-300']
        self.assertEqual(
            case['expected_frames'],
            tama.get_frames(
                wav_fname=case['audio_fname'], words_fname=case['words_fname']
            ),
        )

    def test_frame_separation_long_x2(self):
        case = self.cases['long_100-200-300_x2']
        self.assertEqual(
            case['expected_frames'],
            tama.get_frames(
                wav_fname=case['audio_fname'], words_fname=case['words_fname']
            ),
        )

    def test_frame_separation_spoken(self):
        case = self.cases['spoken']
        self.assertEqual(
            case['expected_frames'],
            tama.get_frames(
                wav_fname=case['audio_fname'], words_fname=case['words_fname']
            ),
        )

    def test_calculate_time_series_empty(self):
        case = self.cases['empty']
        self.assertEqual(
            case['F0_MAX_time_series'],
            tama.calculate_time_series(
                feature="F0_MAX",
                frames=case['expected_frames'],
                audio_file=case['audio_fname'],
                extractor="praat",
            ),
        )

    def test_calculate_time_series_silence(self):
        case = self.cases['silence']
        self.assertEqual(
            case['F0_MAX_time_series'],
            tama.calculate_time_series(
                feature="F0_MAX",
                frames=case['expected_frames'],
                audio_file=case['audio_fname'],
                extractor="praat",
            ),
        )

    def test_calculate_time_series_small(self):
        case = self.cases['small']
        np.testing.assert_almost_equal(
            case['F0_MAX_time_series'],
            tama.calculate_time_series(
                feature="F0_MAX",
                frames=case['expected_frames'],
                audio_file=case['audio_fname'],
                extractor="praat",
            ),
        )

    def test_calculate_time_series_long(self):
        case = self.cases['long_100-200-300']
        np.testing.assert_almost_equal(
            case['F0_MAX_time_series'],
            tama.calculate_time_series(
                feature="F0_MAX",
                frames=case['expected_frames'],
                audio_file=case['audio_fname'],
                extractor="praat",
            ),
        )

    def test_calculate_time_series_long_x2(self):
        case = self.cases['long_100-200-300_x2']
        np.testing.assert_almost_equal(
            case['F0_MAX_time_series'],
            tama.calculate_time_series(
                feature="F0_MAX",
                frames=case['expected_frames'],
                audio_file=case['audio_fname'],
                extractor="praat",
            ),
        )

    def test_calculate_time_series_opensmile_spoken(self):
        case = self.cases['spoken']
        np.testing.assert_almost_equal(
            case['F0final_sma_de_maxPos_time_series'],
            tama.calculate_time_series(
                feature="F0final_sma_de_maxPos",
                frames=case['expected_frames'],
                audio_file=case['audio_fname'],
                extractor="opensmile",
            ),
        )

    def test_calculate_time_series_praat_spoken(self):
        case = self.cases['spoken']
        np.testing.assert_almost_equal(
            case['F0_MAX_time_series'],
            tama.calculate_time_series(
                feature="F0_MAX",
                frames=case['expected_frames'],
                audio_file=case['audio_fname'],
                extractor="praat",
            ),
        )

    def test_calculate_sample_correlation_one_empty(self):
        case = self.cases['empty']
        self.assertRaises(
            ValueError,
            tama.calculate_sample_correlation,
            case['F0_MAX_time_series'],
            [1.0, 2.0, 3.0],  # A random non-empty list
            0,
        )

    def test_calculate_sample_correlation_one_silence(self):
        case = self.cases['silence']
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            np.testing.assert_array_equal(
                np.array([np.nan]),
                tama.calculate_sample_correlation(
                    case['F0_MAX_time_series'],
                    [1.0],  # A random non-empty list
                    0,
                ),
            )

    def test_calculate_sample_correlation_one_silence_many_lags(self):
        case = self.cases['silence']
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            np.testing.assert_array_equal(
                np.array([np.nan, np.nan, np.nan, np.nan, np.nan]),
                tama.calculate_sample_correlation(
                    case['F0_MAX_time_series'],
                    [1.0],  # A random non-empty list
                    4,
                ),
            )

    def test_calculate_sample_correlation_different_time_series_len(self):
        """
        Both time series must correspond to an equal
        amount of frames
        """
        case = self.cases['long_100-200-300']
        self.assertRaises(
            ValueError,
            tama.calculate_sample_correlation,
            case['F0_MAX_time_series'],
            [1.0],  # A random non-empty list
            0,
        )

    def test_calculate_sample_correlation_not_enough_frames(self):
        """
        The numerators should be nan when there
        are less than four non-missing terms
        """
        case = self.cases['long_100-200-300']
        self.assertEqual(
            3,
            np.count_nonzero(
                np.isnan(
                    tama.calculate_sample_correlation(
                        case['F0_MAX_time_series'],
                        case['F0_MAX_time_series'],
                        2,
                    ),
                )
            ),
        )

    def test_calculate_sample_correlation_long(self):
        case = self.cases['long_100-200-300_x2']
        np.testing.assert_almost_equal(
            [1.0, 0.034231247, -0.238247676, 0.113874971, np.nan],
            tama.calculate_sample_correlation(
                case['F0_MAX_time_series'],
                case['F0_MAX_time_series'],
                4,
            ),
        )

    def test_calculate_signed_synchrony(self):
        case = self.cases['long_100-200-300_x2']
        np.testing.assert_almost_equal(
            tama.signed_synchrony(
                case['F0_MAX_time_series'],
                case['F0_MAX_time_series'],
                4,
            ),
            1.0,
        )
