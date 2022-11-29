import argparse
from pathlib import Path
from typing import List

from speechalyze import InterPausalUnit
from speechalyze.continuous import (
    TimeSeries,
    calculate_common_support,
    calculate_metric,
)
from speechalyze.utils import get_interpausal_units, print_audio_description

arg_parser = argparse.ArgumentParser(
    description="Return a times series for a speaker for a task"
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
    "-k",
    "--k-neighboors",
    type=str,
    help="Amount of neighboors to approximate with knn",
)
arg_parser.add_argument(
    "-e", "--extractor", type=str, help="Extractor to use for calculating IPUs features"
)
arg_parser.add_argument(
    "-m",
    "--metric",
    type=str,
    help="Entrainment metric from a/p evolution functions to calculate",
)


def main() -> None:
    args = arg_parser.parse_args()

    k: int = int(args.k_neighboors)
    wav_a_fname: Path = Path(args.audio_file_a)
    words_a_fname: Path = Path(args.words_file_a)
    ipus_a: List[InterPausalUnit] = get_interpausal_units(words_a_fname)
    print(f"Amount of IPUs of speaker A: {len(ipus_a)}")
    print_audio_description("A", wav_a_fname)

    wav_b_fname: Path = Path(args.audio_file_b)
    words_b_fname: Path = Path(args.words_file_b)
    ipus_b: List[InterPausalUnit] = get_interpausal_units(words_b_fname)
    print(f"Amount of IPUs of speaker B: {len(ipus_b)}")
    print_audio_description("B", wav_b_fname)

    for ipu in ipus_a:
        ipu.calculate_features(
            audio_file=wav_a_fname,
            pitch_gender=args.pitch_gender_a,
            extractor=args.extractor,
        )

    for ipu in ipus_b:
        ipu.calculate_features(
            audio_file=wav_b_fname,
            pitch_gender=args.pitch_gender_b,
            extractor=args.extractor,
        )

    time_series_a: List[float] = TimeSeries(
        interpausal_units=ipus_a,
        feature=args.feature,
        method='knn',
        k=k,
    )

    time_series_b: List[float] = TimeSeries(
        interpausal_units=ipus_b,
        feature=args.feature,
        method='knn',
        k=k,
    )

    common_start, common_end = calculate_common_support(time_series_a, time_series_b)
    print(f"Common support: {(common_start, common_end)}")
    metric_result: float = calculate_metric(
        args.metric,
        time_series_a,
        time_series_b,
        common_start,
        common_end,
    )
    print(f"{args.metric}: {metric_result}")


if __name__ == "__main__":
    main()
