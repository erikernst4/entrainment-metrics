import argparse
import subprocess
from pathlib import Path
from typing import List

import numpy as np
from scipy.io import wavfile

from frame import Frame, MissingFrame
from inter_pause_unit import InterPauseUnit

arg_parser = argparse.ArgumentParser(
    description="Generate a times series for a speaker for a task"
)
arg_parser.add_argument(
    "-a", "--audio-file", type=str, help="Audio .wav file for a speaker for a task"
)
arg_parser.add_argument(
    "-w", "--words-file", type=str, help=".words file for a speaker for a task"
)
arg_parser.add_argument(
    "-f", "--feature", type=str, help="Feature to calculate time series"
)


def get_inter_pause_units(words_fname: Path) -> List[InterPauseUnit]:
    """
    Return a list of IPUs given a Path to a .word file
    """
    inter_pause_units: List[InterPauseUnit] = []
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
                inter_pause_units.append(InterPauseUnit(IPU_start, last_end))
                IPU_started = False
        if IPU_start and last_end:  # Last IPU if existent
            inter_pause_units.append(InterPauseUnit(IPU_start, last_end))

    return inter_pause_units


def is_frame_in_inter_pause_unit(inter_pause_unit, frame_start, frame_end):
    res = False

    IPU_start, IPU_end = inter_pause_unit
    max_start = max(IPU_start, frame_start)
    min_end = min(IPU_end, frame_end)

    if max_start < min_end:
        res = True
    return res


def inter_pause_units_inside_interval(inter_pause_units, interval_start, interval_end):
    # POSSIBLE TO-DO: make a logorithmic search
    IPUs = []
    for inter_pause_unit in inter_pause_units:
        if is_frame_in_inter_pause_unit(inter_pause_unit, interval_start, interval_end):
            IPUs.append(inter_pause_unit)
    return IPUs


def separate_frames(
    inter_pause_units: List[InterPauseUnit], data: np.ndarray, samplerate: int
):
    FRAME_LENGHT: int = 16 * samplerate
    TIME_STEP: int = 8 * samplerate

    frames: List[Frame] = []
    audio_length: int = data.shape[0]

    frame_start, frame_end = 0, FRAME_LENGHT
    while frame_start < audio_length:
        # Convert frame ends to seconds
        frame_start_in_s: float = frame_start / samplerate
        frame_end_in_s: float = frame_end / samplerate

        IPUs_inside_frame: List[InterPauseUnit] = inter_pause_units_inside_interval(
            inter_pause_units, frame_start_in_s, frame_end_in_s
        )

        frame = None
        if IPUs_inside_frame:
            frame = Frame(
                start=frame_end_in_s,
                end=frame_end_in_s,
                is_missing=False,
                inter_pause_units=IPUs_inside_frame,
            )
        else:
            # A particular frame could contain no IPUs, in which case its a/p feature values are considered ‘missing’
            frame = MissingFrame(
                start=frame_end_in_s,
                end=frame_end_in_s,
            )

        frames.append(frame)

        frame_start += TIME_STEP
        frame_end += TIME_STEP
        if frame_end > audio_length:
            frame_end = audio_length - 1

    return frames


def calculate_duration_sum(inter_pause_units):
    res = 0
    for IPU_start, IPU_end in inter_pause_units:
        res += IPU_end - IPU_start
    return res


def calculate_features_for_IPU(inter_pause_unit, audio_file):
    IPU_start, IPU_end = inter_pause_unit
    result = subprocess.run(
        [
            'praat',
            './praat_scripts/extractStandardAcoustics.praat',
            '../' + audio_file,
            str(IPU_start),
            str(IPU_end),
            '75',
            '500',
        ],
        stdout=subprocess.PIPE,
        check=True,
    )
    output_lines = result.stdout.decode().rstrip().splitlines()
    features_results = {}
    for line in output_lines:
        feature, value = line.split(":")
        features_results[feature] = float(value)
    # print(f"Feature results for IPU from {IPU_start} to {IPU_end}")
    # print(features_results)
    return features_results


def calculate_time_series(feature, frames, audio_file):
    time_series = []
    for frame in frames:
        if not frame["Is_missing"]:
            IPUs_duration_weighten_mean_values = []
            IPUs_duration_sum = calculate_duration_sum(frame["IPUs"])
            for inter_pause_unit in frame["IPUs"]:
                IPU_start, IPU_end = inter_pause_unit
                IPU_features_results = calculate_features_for_IPU(
                    inter_pause_unit, audio_file
                )
                IPU_duration = IPU_end - IPU_start
                IPU_duration_weighten_mean_value = (
                    IPU_features_results[feature] * IPU_duration
                ) / IPUs_duration_sum
                IPUs_duration_weighten_mean_values.append(
                    IPU_duration_weighten_mean_value
                )
            time_series.append(sum(IPUs_duration_weighten_mean_values))
        else:
            time_series.append(np.nan)
    return time_series


def main() -> None:
    args = arg_parser.parse_args()

    wav_fname: Path = Path(args.audio_file)
    samplerate, data = wavfile.read(wav_fname)
    print(f"Samplerate: {samplerate}")
    print(f"Audio data shape: {data.shape}")
    print(f"Audio data dtype: {data.dtype}")
    print(f"min, max: {data.min()}, {data.max()}")
    print(f"Lenght: {data.shape[0]/samplerate} s")

    words_fname: Path = Path(args.words_file)
    inter_pause_units: List[InterPauseUnit] = get_inter_pause_units(words_fname)
    print(f"Amount of IPUs: {len(inter_pause_units)}")

    frames = separate_frames(inter_pause_units, data, samplerate)
    print(f"Amount of frames: {len(frames)}")
    print(f"Frames: {frames}")

    time_series = calculate_time_series(args.feature, frames, args.audio_file)
    print(f"Time series: {time_series}")


if __name__ == "__main__":
    main()
