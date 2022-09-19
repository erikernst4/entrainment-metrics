from pathlib import Path
from typing import List

from scipy.io import wavfile

from interpausal_unit import InterPausalUnit


def get_interpausal_units(words_fname: Path) -> List[InterPausalUnit]:
    """
    Return a list of IPUs given a Path to a .word file
    """
    interpausal_units: List[InterPausalUnit] = []
    with open(words_fname, encoding="utf-8", mode="r") as word_file:
        IPU_started: bool = False
        IPU_start: float = 0.0
        last_end: float = 0.0
        while line := word_file.readline().rstrip():  # Efficient reading
            start, end, word = line.split()
            word_start, word_end = float(start), float(end)
            if not IPU_started and word == "#":
                IPU_start = 0.0
                last_end = 0.0
            elif not IPU_started and word != "#":
                IPU_start = word_start
                last_end = word_end
                IPU_started = True
            elif IPU_started and word != "#":
                last_end = word_end
            elif IPU_started and word == "#":
                interpausal_units.append(InterPausalUnit(IPU_start, last_end))
                IPU_started = False
        if IPU_start and last_end:  # Last IPU if existent
            interpausal_units.append(InterPausalUnit(IPU_start, last_end))

    return interpausal_units


def print_audio_description(speaker: str, wav_fname: Path) -> None:
    samplerate, data = wavfile.read(wav_fname)
    print("----------------------------------------")
    print(f"Audio from speaker {speaker}")
    print(f"Samplerate: {samplerate}")
    print(f"Audio data shape: {data.shape}")
    print(f"Audio data dtype: {data.dtype}")
    print(f"min, max: {data.min()}, {data.max()}")
    print(f"Lenght: {data.shape[0]/samplerate} s")
    print("----------------------------------------")
