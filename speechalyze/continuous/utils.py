from typing import Optional

import matplotlib.pyplot as plt

from . import TimeSeries


def plot_time_series(
    time_series_a: TimeSeries,
    time_series_b: TimeSeries,
    start: Optional[float] = None,
    end: Optional[float] = None,
    granularity: Optional[float] = None,
    plot_ipus: Optional[bool] = None,
    legend: Optional[bool] = None,
    time_series_a_name: Optional[str] = None,
    time_series_b_name: Optional[str] = None,
    save_fname: Optional[str] = None,
    **kwargs
):
    """
    Plot the predictions of both TimeSeries between
    the given start and end, and with the given granularity.

    Parameters
    ----------
    time_series_a: TimeSeries
        One of the two TimeSeries to plot.
    time_series_b: TimeSeries
        The other TimeSeries to plot.
    start: Optional[float]
        A starting point in time to predict. Default is self.start()
    end: Optional[float]
        An ending point in time to predict. Default is self.end()
    granularity: Optional[float]
        The step in time in which to predict from the time series. Default is 0.01
    plot_ipus: Optional[bool]
        Whether to plot also the InterPausalUnits feature values. Default is True.
    legend: Optional[bool]
        Whether to display a legend. Default is True.
    time_series_a_name: Optional[str]
        The name for the first TimeSeries passed as argument. Default is 'time_series_a'.
    time_series_b_name: Optional[str]
        The name for the second TimeSeries passed as argument. Default is 'time_series_b'.
    save_fname: Optional[str]
        The fname to pass to plt.savefig(). If provided the plot will be saved.
    """
    if (
        legend is None
        or time_series_a_name is not None
        or time_series_b_name is not None
    ):
        legend = True

    if time_series_a_name is None:
        time_series_a_name = "time_series_a"

    if time_series_b_name is None:
        time_series_b_name = "time_series_b"

    time_series_a.plot(
        start=start,
        end=end,
        granularity=granularity,
        plot_ipus=plot_ipus,
        color="C0",
        show=False,
        label=time_series_a_name,
        **kwargs
    )

    time_series_b.plot(
        start=start,
        end=end,
        granularity=granularity,
        plot_ipus=plot_ipus,
        color="C1",
        show=False,
        label=time_series_b_name,
        **kwargs
    )
    if legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())

    if save_fname is not None:
        plt.savefig(save_fname)

    plt.show()
