from typing import List
from unittest import TestCase

from scipy.io import wavfile

from interpausal_unit import InterPausalUnit
from tama import get_interpausal_units


class TAMATestCase(TestCase):
    def setUp(self):
        self.audio = wavfile.read("./data/200-300-100.wav")

    def test_interpausal_units_separation(self):
        expected_ipus: List[InterPausalUnit] = [
            InterPausalUnit(0.0, 0.4),
            InterPausalUnit(0.8, 1.6),
            InterPausalUnit(2.0, 2.4),
        ]
        result_ipus: List[InterPausalUnit] = get_interpausal_units(
            "./data/200-300-100.words"
        )
        self.assertEqual(len(expected_ipus), len(result_ipus))
        for index, ipu in enumerate(result_ipus):
            self.assertEqual(expected_ipus[index].start, ipu.start)
            self.assertEqual(expected_ipus[index].end, ipu.end)
