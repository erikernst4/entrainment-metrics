import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

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


def first_non_outlier_index(
    ipus_feature_values: List[float],
) -> int:
    res = None
    for index, value in enumerate(ipus_feature_values):
        if not np.isnan(value):
            res = index
            break
    return res  # type: ignore


def last_non_outlier_index(
    ipus_feature_values: List[float],
) -> int:
    reversed_index = first_non_outlier_index(list(reversed(ipus_feature_values)))
    real_index = len(ipus_feature_values) - 1 - reversed_index
    return real_index


def calculate_common_support(
    ipus_a: List[InterPausalUnit],
    ipus_a_feature_values: List[float],
    ipus_b: List[InterPausalUnit],
    ipus_b_feature_values: List[float],
) -> Tuple[float, float]:
    a_first_non_outlier_index = first_non_outlier_index(ipus_a_feature_values)
    b_first_non_outlier_index = first_non_outlier_index(ipus_b_feature_values)

    start_a = ipus_a[a_first_non_outlier_index].start
    start_b = ipus_b[b_first_non_outlier_index].start

    start = max(start_a, start_b)

    a_last_non_outlier_index = last_non_outlier_index(ipus_a_feature_values)
    b_last_non_outlier_index = last_non_outlier_index(ipus_b_feature_values)

    end_a = ipus_a[a_last_non_outlier_index].end
    end_b = ipus_b[b_last_non_outlier_index].end

    end = min(end_a, end_b)

    return start, end


def get_interpausal_units_middle_points_in_time(
    interpausal_units: List[InterPausalUnit],
) -> List[float]:
    ipus_middle_point_in_time: List[float] = []
    for ipu in interpausal_units:
        ipu_middle_point_in_time = (ipu.start + ipu.end) / 2
        ipus_middle_point_in_time.append(ipu_middle_point_in_time)
    return ipus_middle_point_in_time


def get_interpausal_units_feature_values(
    feature: str,
    interpausal_units: List[InterPausalUnit],
    audio_file: Path,
    extractor: str,
    pitch_gender: Optional[str] = None,
) -> List[float]:
    ipus_feature_values: List[float] = []
    for ipu in interpausal_units:
        ipu_feature_value = ipu.calculate_features(audio_file, pitch_gender, extractor)[feature]  # type: ignore
        ipus_feature_values.append(ipu_feature_value)
    return ipus_feature_values


def remove_outliers_from_ipus_feature_values(
    ipus_feature_values: List[float],
) -> List[float]:
    """
    Replace outliers with np.nan
    """
    MAX_DEVIATIONS = 3
    array = np.array(ipus_feature_values)
    mean = np.mean(array)
    standard_deviation = np.std(array)
    distance_from_mean = abs(array - mean)
    outlier = distance_from_mean > MAX_DEVIATIONS * standard_deviation
    array[outlier] = np.nan
    return array.tolist()


def calculate_k_nearest_neighboors_from_index(
    index: int,
    k: int,
    ipus_middle_point_in_time: List[float],
    ipus_feature_values: List[float],
) -> Tuple[int, List[float]]:
    k_nearest_neighboors: List[float] = [ipus_feature_values[index]]
    left_index: int = index - 1
    right_index: int = index + 1
    # Add one by one the closest till having k-neighboors
    while (
        len(k_nearest_neighboors) < k
        and left_index >= 0
        and right_index < len(ipus_middle_point_in_time)
    ):
        distance_with_the_left_neighboor: float = (
            ipus_middle_point_in_time[index] - ipus_middle_point_in_time[left_index]
        )
        distance_with_the_right_neighboor: float = (
            ipus_middle_point_in_time[right_index] - ipus_middle_point_in_time[index]
        )
        if distance_with_the_left_neighboor < distance_with_the_right_neighboor:
            k_nearest_neighboors.insert(0, ipus_feature_values[left_index])
            left_index -= 1
        else:
            k_nearest_neighboors.append(ipus_feature_values[right_index])
            right_index += 1

    first_neighboor_index = left_index + 1
    # If there are not k neighboors yet, concatenate the ones left
    if len(k_nearest_neighboors) < k:
        neighboors_left_to_add: int = k - len(k_nearest_neighboors)
        if left_index >= 0:
            first_neighboor_index = left_index - neighboors_left_to_add + 1
            k_nearest_neighboors = (
                ipus_feature_values[first_neighboor_index : left_index + 1]
                + k_nearest_neighboors
            )
        else:
            k_nearest_neighboors = (
                k_nearest_neighboors
                + ipus_feature_values[
                    right_index : right_index + neighboors_left_to_add
                ]
            )
    return first_neighboor_index, k_nearest_neighboors


def calculate_knn_time_series(
    k: int,
    ipus_feature_values: List[float],
    ipus_middle_point_in_time: List[float],
) -> List[float]:
    """
    Generate a time series of the frames values for the feature given


    O(n) being n the amount of interpausal units
    """
    if len(ipus_feature_values) < k:
        raise ValueError("k cannot be smaller than the amount of interpausal units")

    time_series: List[float] = []

    if k == 1:
        return ipus_feature_values

    ipus_feature_values = remove_outliers_from_ipus_feature_values(ipus_feature_values)
    # Initialize neighboors
    k_nearest_neighboors: List[float] = ipus_feature_values[:k]
    first_neighboor_index: int = 0
    time_series.append(np.mean(k_nearest_neighboors))  # type: ignore

    for ipu_index, ipu_middle_point_in_time in enumerate(
        ipus_middle_point_in_time[1:], 1
    ):
        # Calculate k nearest neighboors
        next_neighboor_index = first_neighboor_index + k
        if ipu_index >= next_neighboor_index:
            # If current ipu falls outside of the k_nearest_neighboors, recalculate
            (
                first_neighboor_index,
                k_nearest_neighboors,
            ) = calculate_k_nearest_neighboors_from_index(
                ipu_index, k, ipus_middle_point_in_time, ipus_feature_values
            )
        elif next_neighboor_index < len(ipus_middle_point_in_time):
            # EXPLAIN
            ipu_distance_with_the_first_neighboor = (
                ipu_middle_point_in_time
                - ipus_middle_point_in_time[first_neighboor_index]
            )
            ipu_distance_with_the_next_neighboor = (
                ipus_middle_point_in_time[next_neighboor_index]
                - ipu_middle_point_in_time
            )
            if (
                ipu_distance_with_the_next_neighboor
                < ipu_distance_with_the_first_neighboor
            ):
                # Remove first and append next neighboor
                k_nearest_neighboors = k_nearest_neighboors[1:]
                k_nearest_neighboors.append(ipus_feature_values[next_neighboor_index])
                first_neighboor_index += 1

        time_series.append(np.mean(k_nearest_neighboors))  # type: ignore

    return time_series


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

    ipus_a_middle_points_in_time = get_interpausal_units_middle_points_in_time(ipus_a)
    ipus_a_feature_values = get_interpausal_units_feature_values(
        args.feature, ipus_a, wav_a_fname, args.extractor, args.pitch_gender_a
    )

    ipus_b_middle_points_in_time = get_interpausal_units_middle_points_in_time(ipus_b)
    ipus_b_feature_values = get_interpausal_units_feature_values(
        args.feature, ipus_b, wav_b_fname, args.extractor, args.pitch_gender_b
    )

    common_support: Tuple[float, float] = calculate_common_support(
        ipus_a, ipus_a_feature_values, ipus_b, ipus_b_feature_values
    )
    print(f"Common support: {common_support}")

    time_series_a: List[float] = calculate_knn_time_series(
        k, ipus_a_feature_values, ipus_a_middle_points_in_time
    )
    print("----------------------------------------")
    print(f"Time series of A: {time_series_a}")

    time_series_b: List[float] = calculate_knn_time_series(
        k, ipus_b_feature_values, ipus_b_middle_points_in_time
    )
    print(f"Time series of B: {time_series_b}")
    print("----------------------------------------")


if __name__ == "__main__":
    main()
