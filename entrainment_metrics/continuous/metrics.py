from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np

from entrainment_metrics.continuous import TimeSeries


def calculate_common_support(
    time_series_a: TimeSeries,
    time_series_b: TimeSeries,
) -> Tuple[float, float]:
    """
    Given two times series return the start and end in which
    both TimeSeries are simultaneously defined.


    Parameters
    ----------
    time_series_a: TimeSeries
        One of the two TimeSeries to calculate the common support from.
    time_series_b: TimeSeries
        The other TimeSeries to calculate the common support from.

    Returns
    -------
    Tuple[float, float]
        The start and end of the common support respectively.
    """
    common_start: float = max(time_series_a.start(), time_series_b.start())
    common_end: float = min(time_series_a.end(), time_series_b.end())
    return common_start, common_end


def truncate_values(
    values: np.ndarray,
    start: float,
    end: float,
) -> None:
    greater_than_end_values = values > end
    values[greater_than_end_values] = end

    less_than_start_values = values < start
    values[less_than_start_values] = start


def calculate_proximity(
    time_series_a: TimeSeries,
    time_series_b: TimeSeries,
    start: float,
    end: float,
    granularity: float,
) -> float:
    """
    Calculate the proximity value between two times series

    Metric defined in [QUOTE sigdial2020]

    Parameters
    ----------
    time_series_a: TimeSeries
        One of the two TimeSeries to calculate the metric from.
    time_series_b: TimeSeries
        The other TimeSeries to calculate the metric from.
    start: Optional[float]
        A starting point in time to calculate the metric.
    end: Optional[float]
       An ending point in time to calculate the metric.
    granularity: Optional[float]
        The step in time in which to predict from the time series.
    Returns
    -------
    float
        The metric value.
    """

    time_series_values_a = time_series_a.predict_interval(start, end, granularity)
    time_series_values_b = time_series_b.predict_interval(start, end, granularity)

    mean_a = np.mean(time_series_values_a)
    mean_b = np.mean(time_series_values_b)

    return -abs(mean_a - mean_b)  # type: ignore


def calculate_convergence(
    time_series_a: TimeSeries,
    time_series_b: TimeSeries,
    start: float,
    end: float,
    granularity: float,
) -> float:
    """
    Calculate the convergence value between two times series

    Metric defined in [QUOTE sigdial2020]

    Parameters
    ----------
    time_series_a: TimeSeries
        One of the two TimeSeries to calculate the metric from.
    time_series_b: TimeSeries
        The other TimeSeries to calculate the metric from.
    start: Optional[float]
        A starting point in time to calculate the metric.
    end: Optional[float]
       An ending point in time to calculate the metric.
    granularity: Optional[float]
        The step in time in which to predict from the time series.
    Returns
    -------
    float
        The metric value.
    """

    values_to_predict_in_s = np.arange(start, end + granularity, granularity)

    time_series_values_a = time_series_a.predict_interval(start, end, granularity)
    time_series_values_b = time_series_b.predict_interval(start, end, granularity)

    d_t = np.abs(time_series_values_a - time_series_values_b) * -1
    return np.corrcoef(d_t, values_to_predict_in_s)[0, 1]


def calculate_numerator_montecarlo(
    time_series_values_a_crop: np.ndarray,
    time_series_values_b_crop: np.ndarray,
    mean_a: float,
    mean_b: float,
) -> float:
    numerator_not_integrated = np.multiply(
        time_series_values_a_crop - mean_a, time_series_values_b_crop - mean_b
    )
    # Monte Carlo integration, the lenght of the interval is simplified with the denominator
    numerator = np.mean(numerator_not_integrated)
    return numerator  # type: ignore


def calculate_denominator_montecarlo(
    time_series_values_a_crop: np.ndarray,
    time_series_values_b_crop: np.ndarray,
    mean_a: float,
    mean_b: float,
) -> float:  # type: ignore
    square_distance_to_mean_a = np.square(time_series_values_a_crop - mean_a)
    square_distance_to_mean_b = np.square(time_series_values_b_crop - mean_b)

    # Monte Carlo integration, the lenght of the interval is simplified with the numerator
    integral_a = np.mean(square_distance_to_mean_a)
    integral_b = np.mean(square_distance_to_mean_b)

    denominator = np.sqrt(np.multiply(integral_a, integral_b))

    return denominator


def calculate_synchrony_montecarlo(
    time_series_a: TimeSeries,
    time_series_b: TimeSeries,
    start: float,
    end: float,
    granularity: float,
    synchrony_deltas: List[float],
) -> float:
    # Initialized at min absolute value
    res: float = 0.0

    # Precalculate values and means
    time_series_values_a = time_series_a.predict_interval(start, end, granularity)
    time_series_values_b = time_series_b.predict_interval(start, end, granularity)

    mean_a = np.mean(time_series_values_a)
    mean_b = np.mean(time_series_values_b)

    for synchrony_delta in synchrony_deltas:
        # Validate synchrony_delta
        if abs(synchrony_delta) > end - start:
            raise ValueError(f"Synchrony delta bigger than interval {start} to {end}")

        time_series_values_a_crop = deepcopy(time_series_values_a)
        time_series_values_b_crop = deepcopy(time_series_values_b)

        if synchrony_delta > 0:
            values_to_crop = int(synchrony_delta / granularity)
            time_series_values_a_crop = time_series_values_a_crop[values_to_crop:]
            time_series_values_b_crop = time_series_values_b_crop[:-values_to_crop]

        elif synchrony_delta < 0:
            values_to_crop = int(-synchrony_delta / granularity)
            time_series_values_a_crop = time_series_values_a_crop[:-values_to_crop]
            time_series_values_b_crop = time_series_values_b_crop[values_to_crop:]

        numerator = calculate_numerator_montecarlo(
            time_series_values_a_crop, time_series_values_b_crop, mean_a, mean_b  # type: ignore
        )

        denominator = calculate_denominator_montecarlo(
            time_series_values_a_crop, time_series_values_b_crop, mean_a, mean_b  # type: ignore
        )

        actual_res: float = np.divide(numerator, denominator)

        if np.abs(actual_res) > np.abs(res):
            res = actual_res

    return res


def calculate_numerator_trapz(
    time_series_values_a_crop: np.ndarray,
    time_series_values_b_crop: np.ndarray,
    values_to_predict_in_s: np.ndarray,
    mean_a: float,
    mean_b: float,
) -> float:  # type: ignore
    numerator_not_integrated = np.multiply(
        time_series_values_a_crop - mean_a, time_series_values_b_crop - mean_b
    )

    # We integrate for the xs of values_to_predict_b_in_s in order
    # to keep start <= x + synchrony_delta <= end
    # In other words, to keep the integral well defined,
    # evaluating in points inside the domain
    # [see the integral in the paper].

    # If synchrony_delta > 0, values_to_predict_b_in_s is the interval [start, end - synchrony_delta]
    # start <= start + synchrony_delta
    # and
    # end - synchrony_delta + synchrony_delta = end <= end

    # If synchrony_delta < 0, values_to_predict_b_in_s is the interval [start + abs(synchrony_delta), end]
    # start <= start + abs(synchrony_delta) + synchrony_delta = start
    # and
    # end + synchrony_delta <= end

    return np.trapz(numerator_not_integrated, values_to_predict_in_s)


def calculate_denominator_trapz(
    time_series_values_a_crop: np.ndarray,
    time_series_values_b_crop: np.ndarray,
    values_to_predict_a_in_s: np.ndarray,
    values_to_predict_b_in_s: np.ndarray,
    mean_a: float,
    mean_b: float,
) -> float:  # type: ignore

    square_distance_to_mean_a = np.square(time_series_values_a_crop - mean_a)
    square_distance_to_mean_b = np.square(time_series_values_b_crop - mean_b)

    integral_a = np.trapz(square_distance_to_mean_a, values_to_predict_a_in_s)
    integral_b = np.trapz(square_distance_to_mean_b, values_to_predict_b_in_s)

    denominator = np.sqrt(np.multiply(integral_a, integral_b))

    return denominator


def calculate_synchrony_trapz(
    time_series_a: TimeSeries,
    time_series_b: TimeSeries,
    start: float,
    end: float,
    granularity: float,
    synchrony_deltas: List[float],
) -> float:
    # Initialized at min absolute value
    res: float = 0.0

    # Precalculate global means
    time_series_values_a = time_series_a.predict_interval(start, end, granularity)
    time_series_values_b = time_series_b.predict_interval(start, end, granularity)

    mean_a = np.mean(time_series_values_a)
    mean_b = np.mean(time_series_values_b)

    for synchrony_delta in synchrony_deltas:
        # Validate synchrony_delta
        if abs(synchrony_delta) > end - start:
            raise ValueError(f"Synchrony delta bigger than interval {start} to {end}")

        time_series_values_a_crop = deepcopy(time_series_values_a)
        time_series_values_b_crop = deepcopy(time_series_values_b)

        values_to_predict_a_in_s = None
        values_to_predict_b_in_s = None

        if synchrony_delta >= 0:
            values_to_predict_a_in_s = np.arange(
                start + synchrony_delta, end + granularity, granularity
            )
            values_to_predict_b_in_s = np.arange(
                start, end + granularity - synchrony_delta, granularity
            )
            if synchrony_delta > 0:
                values_to_crop = int(synchrony_delta / granularity)
                time_series_values_a_crop = time_series_values_a_crop[values_to_crop:]
                time_series_values_b_crop = time_series_values_b_crop[:-values_to_crop]

        elif synchrony_delta < 0:
            # Crop the other way with abs(synchrony_delta)
            values_to_crop = int(abs(synchrony_delta) / granularity)
            time_series_values_a_crop = time_series_values_a_crop[:-values_to_crop]
            time_series_values_b_crop = time_series_values_b_crop[values_to_crop:]
            values_to_predict_a_in_s = np.arange(
                start, end + granularity - abs(synchrony_delta), granularity
            )
            values_to_predict_b_in_s = np.arange(
                start + abs(synchrony_delta), end + granularity, granularity
            )

        numerator = calculate_numerator_trapz(
            time_series_values_a_crop, time_series_values_b_crop, values_to_predict_b_in_s, mean_a, mean_b  # type: ignore
        )

        denominator = calculate_denominator_trapz(
            time_series_values_a_crop, time_series_values_b_crop, values_to_predict_a_in_s, values_to_predict_b_in_s, mean_a, mean_b  # type: ignore
        )

        actual_res: float = np.divide(numerator, denominator)

        if np.abs(actual_res) > np.abs(res):
            res = actual_res
    return res


def calculate_synchrony(
    time_series_a: TimeSeries,
    time_series_b: TimeSeries,
    start: float,
    end: float,
    granularity: float,
    synchrony_deltas: Optional[List[float]] = None,
    integration_method: Optional[str] = None,
) -> float:
    """
    Calculate the synchrony value between two times series

    Metric defined in [QUOTE sigdial2020]

    Parameters
    ----------
    time_series_a: TimeSeries
        One of the two TimeSeries to calculate the metric from.
    time_series_b: TimeSeries
        The other TimeSeries to calculate the metric from.
    start: Optional[float]
        A starting point in time to calculate the metric.
    end: Optional[float]
       An ending point in time to calculate the metric.
    granularity: Optional[float]
        The step in time in which to predict from the time series.
    integration_method: Optional[str] = None
        The integration method to use. Methods available: "montecarlo" and "trapz"

    Returns
    -------
    float
        The metric value.
    """
    if synchrony_deltas is None:
        synchrony_deltas = [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0]

    if integration_method is None or integration_method == "montecarlo":
        res = calculate_synchrony_montecarlo(
            time_series_a, time_series_b, start, end, granularity, synchrony_deltas
        )
    elif integration_method == "trapz":
        res = calculate_synchrony_trapz(
            time_series_a, time_series_b, start, end, granularity, synchrony_deltas
        )
    else:
        raise ValueError("Not a valid integration_method given")

    return res


def calculate_metric(
    metric: str,
    time_series_a: TimeSeries,
    time_series_b: TimeSeries,
    start: Optional[float] = None,
    end: Optional[float] = None,
    granularity: Optional[float] = None,
    synchrony_deltas: Optional[List[float]] = None,
    integration_method: Optional[str] = None,
) -> float:
    """
    Calculate entrainment metrics given a times series from each speaker

    Metrics avaible: 'proximity', 'convergence' (AKA 'pearson') and 'synchrony'


    Parameters
    ----------
    metric: str
       The metric to be calculated ("synchrony", "proximity", or "convergence")
    time_series_a: TimeSeries
        One of the two TimeSeries to calculate the metric from.
    time_series_b: TimeSeries
        The other TimeSeries to calculate the metric from.
    start: Optional[float]
        A starting point in time to calculate the metric.
    end: Optional[float]
       An ending point in time to calculate the metric.
    granularity: Optional[float]
        The step in time in which to predict from the time series.
    Returns
    -------
    float
        The metric value.
    """
    if granularity is None:
        granularity = 0.01

    if start is None or end is None:
        common_start, common_end = calculate_common_support(
            time_series_a, time_series_b
        )
        if start is None:
            start = common_start
        if end is None:
            end = common_end

    res = None
    metric = metric.lower()
    if metric == "proximity":
        res = calculate_proximity(time_series_a, time_series_b, start, end, granularity)
    elif metric == "pearson" or metric == "convergence":
        res = calculate_convergence(
            time_series_a, time_series_b, start, end, granularity
        )
    elif metric == "synchrony":
        res = calculate_synchrony(
            time_series_a,
            time_series_b,
            start,
            end,
            granularity,
            synchrony_deltas,
            integration_method,
        )
    else:
        raise ValueError("Not a valid metric")
    return res
