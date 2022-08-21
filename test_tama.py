from typing import List, Union
from unittest import TestCase

from scipy.io import wavfile

from frame import Frame, MissingFrame
from interpausal_unit import InterPausalUnit
from tama import get_frames, get_interpausal_units


class TAMATestCase(TestCase):
    def setUp(self):
        self.cases = {
            'empty': {
                'audio': wavfile.read("./data/empty.wav"),
                'expected_ipus': [],
                'expected_frames': [],
            },
            'silence': {
                'audio': wavfile.read("./data/silence.wav"),
                'expected_ipus': [],
                'expected_frames': [MissingFrame(0.0, 0.3999375)],
            },
            'small': {
                'audio': wavfile.read("./data/200-300-100.wav"),
                'expected_ipus': [
                    InterPausalUnit(0.0, 0.4),
                    InterPausalUnit(0.8, 1.6),
                    InterPausalUnit(2.0, 2.4),
                ],
                'expected_frames': [
                    Frame(
                        0.0,
                        2.3999375,
                        False,
                        [
                            InterPausalUnit(0.0, 0.4),
                            InterPausalUnit(0.8, 1.6),
                            InterPausalUnit(2.0, 2.4),
                        ],
                    )
                ],
            },
        }

    def test_interpausal_units_separation_empty(self):
        self.assertEqual(
            get_interpausal_units("./data/empty.words"),
            self.cases['empty']['expected_ipus'],
        )

    def test_interpausal_units_separation_silence(self):
        self.assertEqual(
            get_interpausal_units("./data/silence.words"),
            self.cases['silence']['expected_ipus'],
        )

    def test_interpausal_units_separation_small(self):
        expected = self.cases['small']
        expected_ipus = expected['expected_ipus']

        result_ipus: List[InterPausalUnit] = get_interpausal_units(
            "./data/200-300-100.words"
        )
        self.assertEqual(expected_ipus, result_ipus)

    def test_frame_separation_empty(self):
        expected_frames = self.cases['empty']['expected_frames']
        result_frames: List[Union[Frame, MissingFrame]] = get_frames(
            wav_fname="./data/empty.wav", words_fname="./data/empty.words"
        )
        self.assertEqual(expected_frames, result_frames)

    def test_frame_separation_silence(self):
        expected_frames = self.cases['silence']['expected_frames']
        result_frames: List[Union[Frame, MissingFrame]] = get_frames(
            wav_fname="./data/silence.wav", words_fname="./data/silence.words"
        )
        print(expected_frames[0])
        print(result_frames[0])
        self.assertEqual(expected_frames, result_frames)

    def test_frame_separation_small(self):
        expected_frames = self.cases['small']['expected_frames']
        result_frames: List[Union[Frame, MissingFrame]] = get_frames(
            wav_fname="./data/200-300-100.wav", words_fname="./data/200-300-100.words"
        )
        self.assertEqual(expected_frames, result_frames)
