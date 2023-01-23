from pathlib import Path
from typing import List, Union

import numpy as np
from scipy.io import wavfile

from entrainment_metrics import InterPausalUnit
from entrainment_metrics.tama import Frame, MissingFrame
from entrainment_metrics.utils import get_interpausal_units


def has_interval_intersection_with_interpausal_unit(
    interpausal_unit: InterPausalUnit, interval_start: float, interval_end: float
) -> bool:
    res: bool = False

    max_start: float = max(interpausal_unit.start, interval_start)
    min_end: float = min(interpausal_unit.end, interval_end)

    if max_start < min_end:
        res = True
    return res


def interpausal_units_inside_interval(
    interpausal_units: List[InterPausalUnit], interval_start: float, interval_end: float
) -> List[InterPausalUnit]:
    """
    Return a list of the IPUs that have intersection with the interval given
    """
    # POSSIBLE TO-DO: make a logorithmic search
    IPUs: List[InterPausalUnit] = []
    for interpausal_unit in interpausal_units:
        if has_interval_intersection_with_interpausal_unit(
            interpausal_unit, interval_start, interval_end
        ):
            IPUs.append(interpausal_unit)
    return IPUs


def separate_frames(
    interpausal_units: List[InterPausalUnit], data: np.ndarray, samplerate: int
) -> List[Union[Frame, MissingFrame]]:
    """
    Given an audio data and samplerate, return a list of the frames inside
    """

    FRAME_LENGHT: int = 16 * samplerate
    TIME_STEP: int = 8 * samplerate

    frames: List[Union[Frame, MissingFrame]] = []
    audio_length: int = data.shape[0]

    frame_start, frame_end = 0, FRAME_LENGHT
    while frame_start < audio_length:
        # Truncate frame_end
        if frame_end > audio_length:
            frame_end = audio_length

        # Convert frame ends to seconds
        frame_start_in_s: float = frame_start / samplerate
        frame_end_in_s: float = frame_end / samplerate

        IPUs_inside_frame: List[InterPausalUnit] = interpausal_units_inside_interval(
            interpausal_units, frame_start_in_s, frame_end_in_s
        )

        frame = None
        if IPUs_inside_frame:
            frame = Frame(
                start=frame_start_in_s,
                end=frame_end_in_s,
                is_missing=False,
                interpausal_units=IPUs_inside_frame,
            )
        else:
            # A particular frame could contain no IPUs, in which case its a/p feature values are considered ‘missing’
            frame = MissingFrame(
                start=frame_start_in_s,
                end=frame_end_in_s,
            )

        frames.append(frame)

        frame_start += TIME_STEP
        frame_end += TIME_STEP

    return frames


def get_frames(
    wav_fname: Path,
    words_fname: Path,
) -> List[Union[Frame, MissingFrame]]:
    """
    Return a list of Frames given a Path to a .word file and a .wav file

    The format of the word file must be:
    - For each line
    f'{starting_time} {ending_time} {word}'
    Where starting_time and ending_time are floats

    Parameters
    ----------
    wav_fname: Path
        The path to the wav file
    words_fname: Path
        The path to the words file

    Returns
    -------
    List[Union[Frame, MissingFrame]]
        The frames from the wav file with the InterPausalUnits from the word file.
    """

    samplerate, data = wavfile.read(wav_fname)

    interpausal_units: List[InterPausalUnit] = get_interpausal_units(words_fname)

    frames: List[Union[Frame, MissingFrame]] = separate_frames(
        interpausal_units, data, samplerate
    )

    return frames
