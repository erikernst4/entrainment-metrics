from typing import List, Union
from unittest import TestCase

from scipy.io import wavfile

from frame import Frame, MissingFrame
from interpausal_unit import InterPausalUnit
from tama import get_frames, get_interpausal_units


class TAMATestCase(TestCase):
    def setUp(self):
        self.cases = {
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
            }
        }

    def test_interpausal_units_separation_small(self):
        expected = self.cases['small']
        expected_ipus = expected['expected_ipus']

        result_ipus: List[InterPausalUnit] = get_interpausal_units(
            "./data/200-300-100.words"
        )
        self.assertEqual(expected_ipus, result_ipus)

    def test_frame_separation_one_frame(self):
        expected_frames = self.cases['small']['expected_frames']
        result_frames: List[Union[Frame, MissingFrame]] = get_frames(
            "", "./data/200-300-100.wav", "./data/200-300-100.words"
        )
        self.assertEqual(expected_frames, result_frames)
