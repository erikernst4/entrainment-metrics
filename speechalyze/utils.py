from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

from .interpausal_unit import InterPausalUnit


def get_interpausal_units(words_fname: Path) -> List[InterPausalUnit]:
    """
    Return a list of IPUs given a Path to a .word file

    The format of the file must be:
        - For each line
            f'{starting_time} {ending_time} {word}'
        Where starting_time and ending_time are floats

    Parameters
    ----------
    words_fname: Path
        The path to the words file

    Returns
    -------
    List[InterPausalUnit]
        The InterPausalUnits from the words file.
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


def print_ipus_information(ipus: List[InterPausalUnit], feature: str):
    """
    Print lenght, std, mean, min, max, min start, and max end
    from the list of IPUs.

    Parameters
    ----------
    ipus: List[InterPausalUnit]
        The list of IPUs from which to extract information.
    feature: str
        The feature from which to extract the feature value of each InterPausalUnit.
    """
    start = ipus[0].start
    end = ipus[0].end
    for ipu in ipus:
        if ipu.start < start:
            start = ipu.start
        if ipu.end > end:
            end = ipu.end

    ipus_feature_values = [ipu.feature_value(feature) for ipu in ipus]
    mean = np.mean(ipus_feature_values)
    std = np.std(ipus_feature_values)

    print(f"Amount of IPUs: {len(ipus)}")
    print(f"Std: {std}")
    print(f"Mean: {mean}")
    print(f"Min {feature} value: {np.min(ipus_feature_values)}")
    print(f"Max {feature} value: {np.max(ipus_feature_values)}")
    print(f"Min start: {start}")
    print(f"Max end: {end}")


def plot_ipus(ipus: List[InterPausalUnit], feature: str, **kwargs):
    """
    Plot the IPU's feature values with its corresponding lenght.

    Parameters
    ----------
    ipus: List[InterPausalUnit]
        The list of IPUs from which to extract information.
    feature: str
        The feature from which to extract the feature value of each InterPausalUnit.
    """
    ipus_values = [ipu.feature_value(feature) for ipu in ipus]
    ipus_starts = [ipu.start for ipu in ipus]
    ipus_ends = [ipu.end for ipu in ipus]
    plt.hlines(y=ipus_values, xmin=ipus_starts, xmax=ipus_ends, **kwargs)
    plt.xlabel("Time (seconds)")
    plt.ylabel(feature)
    plt.show()
