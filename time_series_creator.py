import argparse
import subprocess
from pathlib import Path

import numpy as np
from scipy.io import wavfile

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


def get_relevant_frames_utterances(words_fname):
    relevant_frames_utterances = []
    with open(words_fname, encoding="utf-8", mode="r") as word_file:
        rfu_started = False
        rfu_start = 0.0
        last_end = 0.0
        while line := word_file.readline().rstrip():  # Efficient reading
            start, end, word = line.split()
            if not rfu_started and word == "#":
                rfu_start = 0.0
                last_end = 0.0
            elif not rfu_started and word != "#":
                rfu_start = start
                last_end = end
                rfu_started = True
            elif rfu_started and word != "#":
                last_end = end
            elif rfu_started and word == "#":
                rfu_start, rfu_end = float(rfu_start), float(last_end)
                relevant_frames_utterances.append((rfu_start, rfu_end))
                rfu_started = False
        if rfu_start and last_end:  # Last RFU if existent
            relevant_frames_utterances.append((float(rfu_start), float(last_end)))

    return relevant_frames_utterances


def is_frame_in_relevant_frame_utterance(
    relevant_frames_utterance, frame_start, frame_end
):
    res = False

    rfu_start, rfu_end = relevant_frames_utterance
    max_start = max(rfu_start, frame_start)
    min_end = min(rfu_end, frame_end)

    if max_start < min_end:
        res = True
    return res


def relevant_frames_utterances_inside_frame(relevant_frames_utterances, frame):
    # POSSIBLE TO-DO: make a logorithmic search
    rfus = []
    for relevant_frames_utterance in relevant_frames_utterances:
        if is_frame_in_relevant_frame_utterance(
            relevant_frames_utterance, frame["Start"], frame["End"]
        ):
            rfus.append(relevant_frames_utterance)
    return rfus


def separate_frames(relevant_frames_utterances, data, samplerate):
    FRAME_LENGHT = 16 * samplerate
    TIME_STEP = 8 * samplerate

    frames = []
    audio_length = data.shape[0]

    frame_start, frame_end = 0, FRAME_LENGHT
    while frame_start < audio_length:
        # Convert frame ends to seconds
        frame = {}
        frame_start_in_s = frame_start / samplerate
        frame_end_in_s = frame_end / samplerate

        frame["Start"] = frame_start_in_s
        frame["End"] = frame_end_in_s
        rfus_inside_frame = relevant_frames_utterances_inside_frame(
            relevant_frames_utterances, frame
        )
        if rfus_inside_frame:
            frame["Is_missing"] = False
            frame["RFUs"] = rfus_inside_frame
        else:
            # A particular frame could contain no RFUs, in which case its a/p feature values are considered ‘missing’
            frame["Is_missing"] = True

        frames.append(frame)

        frame_start += TIME_STEP
        frame_end += TIME_STEP
        if frame_end > audio_length:
            frame_end = audio_length - 1

    return frames


def calculate_duration_sum(relevant_frames_utterances):
    res = 0
    for rfu_start, rfu_end in relevant_frames_utterances:
        res += rfu_end - rfu_start
    return res


def calculate_features_for_rfu(relevant_frames_utterance, audio_file):
    rfu_start, rfu_end = relevant_frames_utterance
    result = subprocess.run(
        [
            'praat',
            './praat_scripts/extractStandardAcoustics.praat',
            '../' + audio_file,
            str(rfu_start),
            str(rfu_end),
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
    # print(f"Feature results for RFU from {rfu_start} to {rfu_end}")
    # print(features_results)
    return features_results


def calculate_time_series(feature, frames, audio_file):
    time_series = []
    for frame in frames:
        if not frame["Is_missing"]:
            rfus_duration_weighten_mean_values = []
            rfus_duration_sum = calculate_duration_sum(frame["RFUs"])
            for relevant_frames_utterance in frame["RFUs"]:
                rfu_start, rfu_end = relevant_frames_utterance
                rfu_features_results = calculate_features_for_rfu(
                    relevant_frames_utterance, audio_file
                )
                rfu_duration = rfu_end - rfu_start
                rfu_duration_weighten_mean_value = (
                    rfu_features_results[feature] * rfu_duration
                ) / rfus_duration_sum
                rfus_duration_weighten_mean_values.append(
                    rfu_duration_weighten_mean_value
                )
            time_series.append(sum(rfus_duration_weighten_mean_values))
        else:
            time_series.append(np.nan)
    return time_series


def main() -> None:
    args = arg_parser.parse_args()

    wav_fname = Path(args.audio_file)
    samplerate, data = wavfile.read(wav_fname)
    print(f"Samplerate: {samplerate}")
    print(f"Audio data shape: {data.shape}")
    print(f"Audio data dtype: {data.dtype}")
    print(f"min, max: {data.min()}, {data.max()}")
    print(f"Lenght: {data.shape[0]/samplerate} s")

    words_fname = Path(args.words_file)
    relevant_frames_utterances = get_relevant_frames_utterances(words_fname)
    print(f"Amount of RFUs: {len(relevant_frames_utterances)}")

    frames = separate_frames(relevant_frames_utterances, data, samplerate)
    print(f"Amount of frames: {len(frames)}")
    print(f"Frames: {frames}")

    time_series = calculate_time_series(args.feature, frames, args.audio_file)
    print(f"Time series: {time_series}")


if __name__ == "__main__":
    main()
