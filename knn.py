import argparse
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from continuous_time_series import TimeSeries
from interpausal_unit import InterPausalUnit
from utils import get_interpausal_units, print_audio_description

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
arg_parser.add_argument(
    "-sdelta",
    "--synchrony-delta",
    type=str,
    help="Extractor to use for calculating IPUs features",
)


def calculate_common_support(
    time_series_a: TimeSeries,
    time_series_b: TimeSeries,
) -> Tuple[float, float]:
    common_start: float = max(time_series_a.start(), time_series_b.start())
    common_end: float = min(time_series_a.end(), time_series_b.end())
    return common_start, common_end


def calculate_proximity(
    time_series_a: TimeSeries,
    time_series_b: TimeSeries,
    start: float,
    end: float,
    granularity: float,
) -> float:
    values_to_predict_in_s = np.arange(start, end, granularity)
    values_to_predict = values_to_predict_in_s.reshape(-1, 1)

    time_series_values_a = time_series_a.predict(values_to_predict)
    time_series_values_b = time_series_b.predict(values_to_predict)

    mean_a = np.mean(time_series_values_a)
    mean_b = np.mean(time_series_values_b)

    return -abs(mean_a - mean_b)  # type: ignore


def calculate_convergence(
    time_series_a: TimeSeries,
    time_series_b: TimeSeries,
    start: float,
    end: float,
    granularity: float,
) -> np.ndarray:

    values_to_predict_in_s = np.arange(start, end, granularity)
    values_to_predict = values_to_predict_in_s.reshape(-1, 1)

    time_series_values_a = time_series_a.predict(values_to_predict)
    time_series_values_b = time_series_b.predict(values_to_predict)

    d_t = np.abs(time_series_values_a - time_series_values_b) * -1
    return np.corrcoef(d_t, values_to_predict_in_s)[0, 1]


def calculate_synchrony(
    time_series_a: TimeSeries,
    time_series_b: TimeSeries,
    start: float,
    end: float,
    granularity: float,
    synchrony_delta: Optional[float],
) -> np.ndarray:
    if synchrony_delta is None:
        synchrony_delta = 5.0

    values_to_predict_a_in_s = np.arange(start + synchrony_delta, end, granularity)
    values_to_predict_b_in_s = np.arange(start, end - synchrony_delta, granularity)
    values_to_predict_a, values_to_predict_b = values_to_predict_a_in_s.reshape(
        -1, 1
    ), values_to_predict_b_in_s.reshape(-1, 1)

    time_series_values_a = time_series_a.predict(values_to_predict_a)
    time_series_values_b = time_series_b.predict(values_to_predict_b)

    return np.corrcoef(time_series_values_a, time_series_values_b)[0, 1]


def calculate_metric(
    metric: str,
    time_series_a: TimeSeries,
    time_series_b: TimeSeries,
    start: float,
    end: float,
    synchrony_delta: Optional[float] = None,
    granularity: Optional[float] = None,
) -> Any:
    """
    Calculate entrainment metrics given a times series from each speaker

    Metrics avaible: 'proximity', 'pearson'
    """
    if granularity is None:
        granularity = 0.01

    res: Any = None
    metric = metric.lower()
    if metric == "proximity":
        res = calculate_proximity(time_series_a, time_series_b, start, end, granularity)
    elif metric == "pearson" or metric == "convergence":
        res = calculate_convergence(
            time_series_a, time_series_b, start, end, granularity
        )
    elif metric == "synchrony":
        res = calculate_synchrony(
            time_series_a, time_series_b, start, end, granularity, synchrony_delta
        )
    else:
        raise ValueError("Not a valid metric")
    return res


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
    print("----------------------------------------")
    print(f"Time series of A: {time_series_a}")

    time_series_b: List[float] = TimeSeries(
        interpausal_units=ipus_b,
        feature=args.feature,
        method='knn',
        k=k,
    )
    print(f"Time series of B: {time_series_b}")
    print("----------------------------------------")

    common_start, common_end = calculate_common_support(time_series_a, time_series_b)
    print(f"Common support: {(common_start, common_end)}")
    synchrony_delta = None
    if args.synchrony_delta:
        synchrony_delta = float(args.synchrony_delta)
    metric_result: Any = calculate_metric(
        args.metric,
        time_series_a,
        time_series_b,
        common_start,
        common_end,
        synchrony_delta,
    )
    print(f"{args.metric}: {metric_result}")


if __name__ == "__main__":
    main()
