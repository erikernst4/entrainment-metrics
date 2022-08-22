from unittest import TestCase

import numpy as np
from scipy.io import wavfile

from entrainment import calculate_time_series
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
            },
            'silence': {
                'words_fname': "./data/silence.words",
                'audio_fname': "./data/silence.wav",
                'audio': wavfile.read("./data/silence.wav"),
                'expected_ipus': [],
                'expected_frames': [MissingFrame(0.0, 0.4)],
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
                            InterPausalUnit(0.0, 0.4),
                            InterPausalUnit(0.8, 1.6),
                            InterPausalUnit(2.0, 2.4),
                        ],
                    )
                ],
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
                            InterPausalUnit(0.0, 4.0),
                            InterPausalUnit(8.0, 12.0),
                        ],
                    ),
                    Frame(
                        8.0,
                        24.0,
                        False,
                        [
                            InterPausalUnit(8.0, 12.0),
                            InterPausalUnit(16.0, 24.0),
                        ],
                    ),
                    Frame(
                        16.0,
                        24.0,
                        False,
                        [
                            InterPausalUnit(16.0, 24.0),
                        ],
                    ),
                ],
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

    def test_calulate_time_series_empty(self):
        case = self.cases['empty']
        self.assertEqual(
            [],
            calculate_time_series(
                "F0_MAX", case['expected_frames'], case['audio_fname']
            ),
        )

    def test_calulate_time_series_silence(self):
        case = self.cases['silence']
        self.assertEqual(
            [np.nan],
            calculate_time_series(
                "F0_MAX", case['expected_frames'], case['audio_fname']
            ),
        )
