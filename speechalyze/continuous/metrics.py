from typing import Optional, Tuple

import numpy as np
from scipy.integrate import quad as integrate

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


def calculate_synchrony_numerator(
    time_series_a: TimeSeries,
    time_series_b: TimeSeries,
    synchrony_delta: float,
    mean_a: float,  # type: ignore
    mean_b: float,  # type: ignore
    start: float,
    end: float,
) -> float:
    numerator: float = integrate(
        lambda x: np.multiply(
            time_series_a.predict(x + synchrony_delta) - mean_a,
            time_series_b.predict(x) - mean_b,
        ),
        start,
        end,
    )[0]
    return numerator


def calculate_synchrony_denominator(
    time_series_a: TimeSeries,
    time_series_b: TimeSeries,
    synchrony_delta: float,
    mean_a: float,
    mean_b: float,
    start: float,
    end: float,
) -> float:
    square_distance_to_mean_a = integrate(
        lambda x: np.square(time_series_a.predict(x + synchrony_delta) - mean_a),
        start,
        end,
    )[0]
    square_distance_to_mean_b = integrate(
        lambda x: np.square(time_series_b.predict(x) - mean_b), start, end
    )[0]

    denominator = np.sqrt(
        np.multiply(square_distance_to_mean_a, square_distance_to_mean_b)
    )

    return denominator


def calculate_synchrony(
    time_series_a: TimeSeries,
    time_series_b: TimeSeries,
    start: float,
    end: float,
    granularity: float,
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
    # Initialized at min float
    res: float = np.finfo(np.float64).min  # type: ignore

    # Precalculate means
    time_series_values_a, time_series_values_b = calculate_time_series_values(
        time_series_a, time_series_b, start, end, granularity
    )

    mean_a = np.mean(time_series_values_a)
    mean_b = np.mean(time_series_values_b)

    for synchrony_delta in [-15, -10, -5, 0, 5, 10, 15]:
        denominator: float = calculate_synchrony_denominator(
            time_series_a, time_series_b, synchrony_delta, mean_a, mean_b, start, end  # type: ignore
        )
        numerator: float = calculate_synchrony_numerator(
            time_series_a, time_series_b, synchrony_delta, mean_a, mean_b, start, end  # type: ignore
        )

        actual_res: float = np.divide(numerator, denominator)

        if actual_res > res:
            res = actual_res

    return res


def calculate_metric(
    metric: str,
    time_series_a: TimeSeries,
    time_series_b: TimeSeries,
    start: Optional[float] = None,
    end: Optional[float] = None,
    granularity: Optional[float] = None,
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
        res = calculate_synchrony(time_series_a, time_series_b, start, end, granularity)
    else:
        raise ValueError("Not a valid metric")
    return res
