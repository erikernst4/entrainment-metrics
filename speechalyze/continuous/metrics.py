from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np

from speechalyze.continuous import TimeSeries


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


def calculate_time_series_values(
    time_series_a: TimeSeries,
    time_series_b: TimeSeries,
    start: float,
    end: float,
    granularity: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Predict the values of two times series between the given
    start and end, and with the given granularity.


    Parameters
    ----------
    time_series_a: TimeSeries
        One of the two TimeSeries to predict from.
    time_series_b: TimeSeries
        The other TimeSeries to predict from.
    start: Optional[float]
        A starting point in time to predict.
    end: Optional[float]
       An ending point in time to predict.
    granularity: Optional[float]
        The step in time in which to predict from the time series.
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The arrays of values predicted corresponded with the two TimeSeries given respectively.
    """

    values_to_predict_in_s = np.arange(start, end + granularity, granularity)
    truncate_values(values_to_predict_in_s, start, end)
    values_to_predict = values_to_predict_in_s.reshape(-1, 1)

    time_series_values_a = time_series_a.predict(values_to_predict)
    time_series_values_b = time_series_b.predict(values_to_predict)

    return time_series_values_a, time_series_values_b


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

    time_series_values_a, time_series_values_b = calculate_time_series_values(
        time_series_a, time_series_b, start, end, granularity
    )

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

    time_series_values_a, time_series_values_b = calculate_time_series_values(
        time_series_a, time_series_b, start, end, granularity
    )

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
    synchrony_deltas: Optional[List[float]] = None,
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
    Returns
    -------
    float
        The metric value.
    """
    if synchrony_deltas is None:
        synchrony_deltas = [0.0, 5.0, 10.0, 15.0]
        # synchrony_deltas = [-15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0]

    # Initialized at min absolute value
    res: float = 0.0

    # Precalculate values and means
    time_series_values_a, time_series_values_b = calculate_time_series_values(
        time_series_a, time_series_b, start, end, granularity
    )

    mean_a = np.mean(time_series_values_a)
    mean_b = np.mean(time_series_values_b)

    for synchrony_delta in synchrony_deltas:
        # Validate synchrony_delta
        if synchrony_delta > end - start:
            raise ValueError(f"Synchrony delta bigger than interval {start} to {end}")

        time_series_values_a_crop = deepcopy(time_series_values_a)
        time_series_values_b_crop = deepcopy(time_series_values_b)

        if synchrony_delta != 0:
            values_to_crop = int(synchrony_delta / granularity)
            time_series_values_a_crop = time_series_values_a_crop[values_to_crop:]
            time_series_values_b_crop = time_series_values_b_crop[:-values_to_crop]

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


def calculate_metric(
    metric: str,
    time_series_a: TimeSeries,
    time_series_b: TimeSeries,
    start: Optional[float] = None,
    end: Optional[float] = None,
    granularity: Optional[float] = None,
    synchrony_deltas: Optional[List[float]] = None,
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
        res = calculate_synchrony_montecarlo(
            time_series_a, time_series_b, start, end, granularity, synchrony_deltas
        )
    else:
        raise ValueError("Not a valid metric")
    return res
