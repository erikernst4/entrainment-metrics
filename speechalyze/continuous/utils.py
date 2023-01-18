from typing import Optional

import matplotlib.pyplot as plt

from . import TimeSeries


def plot_time_series(
    time_series_a: TimeSeries,
    time_series_b: TimeSeries,
    plot_ipus: Optional[bool] = None,
    granularity: Optional[float] = None,
    legend: Optional[bool] = None,
    time_series_a_name: Optional[str] = None,
    time_series_b_name: Optional[str] = None,
    **kwargs
):
    if legend is None:
        legend = True

    if time_series_a_name is None:
        time_series_a_name = "time_series_a"

    if time_series_b_name is None:
        time_series_b_name = "time_series_b"

    time_series_a.plot(
        plot_ipus,
        granularity,
        color="C0",
        show=False,
        label=time_series_a_name,
        **kwargs
    )
    time_series_b.plot(
        plot_ipus,
        granularity,
        color="C1",
        show=False,
        label=time_series_b_name,
        **kwargs
    )
    if legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
    plt.show()
