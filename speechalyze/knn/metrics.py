from typing import Optional, Tuple

import numpy as np

from speechalyze.knn import TimeSeries


def calculate_common_support(
    time_series_a: TimeSeries,
    time_series_b: TimeSeries,
) -> Tuple[float, float]:
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
    values_to_predict_in_s = np.arange(start, end + granularity, granularity)
    truncate_values(values_to_predict_in_s, start, end)
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
) -> float:

    values_to_predict_in_s = np.arange(start, end + granularity, granularity)
    truncate_values(values_to_predict_in_s, start, end)
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
) -> float:
    # Initialized at min float
    res: float = np.finfo(np.float64).min  # type: ignore
    for synchrony_delta in [-15, -10, -5, 0, 5, 10, 15]:
        values_to_predict_a_in_s = np.arange(
            start + synchrony_delta, end + granularity, granularity
        )
        values_to_predict_b_in_s = np.arange(
            start, end - synchrony_delta + granularity, granularity
        )
        truncate_values(values_to_predict_a_in_s, start, end)
        truncate_values(values_to_predict_b_in_s, start, end)
        values_to_predict_a, values_to_predict_b = (
            values_to_predict_a_in_s.reshape(-1, 1),
            values_to_predict_b_in_s.reshape(-1, 1),
        )

        time_series_values_a = time_series_a.predict(values_to_predict_a)
        time_series_values_b = time_series_b.predict(values_to_predict_b)

        actual_res = np.corrcoef(time_series_values_a, time_series_values_b)[0, 1]

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

    Metrics avaible: 'proximity', 'pearson'
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
