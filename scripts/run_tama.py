import argparse
from pathlib import Path
from typing import List, Union

from speechalyze import print_audio_description, tama

arg_parser = argparse.ArgumentParser(
    description="Generate a times series for a speaker for a task"
)
arg_parser.add_argument(
    "-a", "--audio-file-a", type=str, help="Audio .wav file for a speaker A"
)
arg_parser.add_argument(
    "-b", "--audio-file-b", type=str, help="Audio .wav file for a speaker B"
)
arg_parser.add_argument(
    "-wa", "--words-file-a", type=str, help=".words file for a speaker A"
)
arg_parser.add_argument(
    "-wb", "--words-file-b", type=str, help=".words file for a speaker B"
)
arg_parser.add_argument(
    "-f", "--feature", type=str, help="Feature to calculate time series"
)
arg_parser.add_argument(
    "-ga", "--pitch-gender-a", type=str, help="Gender of the pitch of speaker A"
)
arg_parser.add_argument(
    "-gb", "--pitch-gender-b", type=str, help="Gender of the pitch of speaker B"
)
arg_parser.add_argument(
    "-l",
    "--lags",
    type=str,
    help="Variation of lags to calculate Sample cross-correlation",
)
arg_parser.add_argument(
    "-e", "--extractor", type=str, help="Extractor to use for calculating IPUs features"
)


def main() -> None:
    args = arg_parser.parse_args()

    wav_a_fname: Path = Path(args.audio_file_a)
    words_a_fname: Path = Path(args.words_file_a)
    frames_a: List[Union[tama.Frame, tama.MissingFrame]] = tama.get_frames(
        wav_a_fname, words_a_fname
    )
    print(f"Amount of frames of speaker A: {len(frames_a)}")

    print_audio_description("A", wav_a_fname)

    wav_b_fname: Path = Path(args.audio_file_b)
    words_b_fname: Path = Path(args.words_file_b)
    frames_b: List[Union[tama.Frame, tama.MissingFrame]] = tama.get_frames(
        wav_b_fname, words_b_fname
    )
    print_audio_description("B", wav_b_fname)
    print(f"Amount of frames of speaker B: {len(frames_b)}")

    if len(frames_a) != len(frames_b):
        raise ValueError("The amount of frames of each speaker is different")

    time_series_a: List[float] = tama.calculate_time_series(
        args.feature, frames_a, wav_a_fname, args.extractor, args.pitch_gender_a
    )
    print("----------------------------------------")
    print(f"Time series of A: {time_series_a}")

    time_series_b: List[float] = tama.calculate_time_series(
        args.feature, frames_b, wav_b_fname, args.extractor, args.pitch_gender_b
    )
    print(f"Time series of B: {time_series_b}")
    print("----------------------------------------")

    print("Sample cross-correlation")
    sample_cross_correlations: List[float] = tama.calculate_sample_correlation(
        time_series_a, time_series_b, int(args.lags)
    )
    print(f"Correlations with lag from 0 to {args.lags}: {sample_cross_correlations}")


if __name__ == "__main__":
    main()
