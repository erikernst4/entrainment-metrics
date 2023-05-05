import warnings
from copy import deepcopy
from math import isnan
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from entrainment_metrics import InterPausalUnit


class TimeSeries:
    """The evolution of an acoustic-prosodic feature
    value in time.


    Parameters
    ----------
    feature: str
        The feature to get the value from each InterPausalUnit

    interpausal_units: List[InterPausalUnit]
        An ordered list of InterPausalUnit's

    method: str
        The method to be used to predict

    k: Optional[int]
        The amount of neighbors to use in KNeighborsRegressor

    MAX_DEVIATIONS: Optional[int]
        The amount of deviation to define an outlier

    """

    def __init__(
        self,
        feature: str,
        interpausal_units: List[InterPausalUnit],
        method: str,
        k: Optional[int] = None,
        MAX_DEVIATIONS: Optional[int] = None,
        **kwargs,
    ) -> None:
        #: The InterPausalUnits of the TimeSeries.
        self.ipus: List[InterPausalUnit] = self._clean_ipus(interpausal_units, feature)

        #: The feature to get the value from each InterPausalUnit.
        self.feature: str = feature

        #: The feature values of each ipu.
        self.ipus_feature_values: np.ndarray = (
            self._get_interpausal_units_feature_values()
        )

        self.outliers = None

        # Removes IPUs with an outlier feature value and their values in ipus_feature_values
        self._prepare_data(MAX_DEVIATIONS)

        if method == "knn":
            if k is None:
                k = 7

            if len(interpausal_units) < k:
                raise ValueError(
                    "k cannot be bigger than the amount of interpausal units, default k is 7"
                )

            self.model = KNeighborsRegressor(n_neighbors=k, **kwargs)

            # Define X without outliers IPUs
            X = self._get_middle_points_in_time()
            X = X.reshape(-1, 1)

            self.model.fit(X, self.ipus_feature_values)
        else:
            # Here is some space to build your own model!
            raise ValueError("Model to be implemented")

    def __repr__(self):
        return f"TimeSeries(start={self.start()}, end={self.end()}, feature={self.feature}, interpausal_units={self.ipus})"

    def _clean_ipus(
        self,
        interpausal_units: List[InterPausalUnit],
        feature: str,
    ) -> List[InterPausalUnit]:
        ipus = deepcopy(interpausal_units)
        ipus_wo_feature = [
            ipu
            for ipu in ipus
            if ipu.feature_value(feature) is None or isnan(ipu.feature_value(feature))
        ]
        if ipus_wo_feature:
            warnings.warn(
                f"""InterPausalUnit with None or NaN value: the following InterPausalUnit's do not have a value for {feature}:
                    {ipus_wo_feature}
                    Default behaviour discards this InterPausalUnit/s
                """
            )
        return [ipu for ipu in ipus if ipu not in ipus_wo_feature]

    def _get_interpausal_units_feature_values(
        self,
    ) -> np.ndarray:
        """
        Returns a list with the feature value for each IPU.
        """
        return np.array([ipu.feature_value(self.feature) for ipu in self.ipus])

    def _prepare_data(
        self,
        MAX_DEVIATIONS: Optional[int] = None,
    ) -> None:
        """
        Removes IPUs with an outlier feature value and their values in ipus_feature_values

        Outliers are values with a distance from the mean greater than MAX_DEVIATIONS times the standard deviation.
        WARNING: self.ipus and self.ipus_feature_values may be modified
        """
        if MAX_DEVIATIONS is None:
            MAX_DEVIATIONS = 3

        mean: np.float64 = np.mean(self.ipus_feature_values)
        standard_deviation: np.float64 = np.std(self.ipus_feature_values)
        distance_from_mean: np.ndarray = abs(self.ipus_feature_values - mean)
        not_outlier: np.ndarray = (
            distance_from_mean < MAX_DEVIATIONS * standard_deviation
        )
        # Update IPUs to not outlier ipus and its respective feature values
        self.outliers = np.array(self.ipus)[~not_outlier].size
        self.ipus = np.array(self.ipus)[not_outlier].tolist()

        self.ipus_feature_values = self.ipus_feature_values[not_outlier]

    def _get_middle_points_in_time(
        self,
    ) -> np.ndarray:
        """
        Returns a list with the middle point in time of each IPU.
        """
        return np.array([(ipu.start + ipu.end) / 2 for ipu in self.ipus])

    def start(
        self,
    ) -> float:
        """
        Returns the starting point in time in which
        the TimeSeries is defined
        """
        return (self.ipus[0].end + self.ipus[0].start) / 2

    def end(
        self,
    ) -> float:
        """
        Returns the ending point in time in which
        the TimeSeries is defined
        """
        return (self.ipus[-1].end + self.ipus[-1].start) / 2

    def predict(
        self,
        X,
    ) -> np.ndarray:
        """
        Given a point or an array of points in time,
        predict the TimeSeries value for the feature
        of the TimeSeries.


        Parameters
        ----------
        X: float, list or np.ndarray
            A point or an array/list of points in time.

        Returns
        -------
        np.ndarray
            The predicted value/s for the point/s in time given.
        """
        # Convert float to expected predict type
        if isinstance(X, float):
            X = [X]

        # Validate input
        if isinstance(X, list) or (isinstance(X, np.ndarray) and X.ndim == 1):
            X = np.array(X).reshape(-1, 1)
        else:
            raise ValueError(
                """Invalid input: the value/s to predict must be a float or
                a 1 dimentional list or numpy array with the points in time to predict.
                """
            )

        for x in X:
            if x > self.end():
                warnings.warn(
                    f"""Out of bounds {x}: A value in X is greater than TimeSeries end.
                    Remember the end of a TimeSeries is the middle point of the last non-outlier IPU.
                """
                )
            if x < self.start():
                warnings.warn(
                    f"""Out of bounds {x}: A value in X is smaller than TimeSeries start.
                    Remember the start of a TimeSeries is the middle point of the first non-outlier IPU.
                """
                )
        return self.model.predict(X)

    def predict_interval(
        self,
        start: Optional[float] = None,
        end: Optional[float] = None,
        granularity: Optional[float] = None,
    ) -> np.ndarray:
        """
        Predict the values of the times series between the given
        start and end, and with the given granularity.


        Parameters
        ----------
        start: Optional[float]
            A starting point in time to predict. Default is self.start()
        end: Optional[float]
            An ending point in time to predict. Default is self.end()
        granularity: Optional[float]
            The step in time in which to predict from the time series. Default is 0.01
        Returns
        -------
        np.ndarray
            The array with the values predicted.
        """

        if start is None:
            start = self.start()

        if end is None:
            end = self.end()

        if granularity is None:
            granularity = 0.01

        # Prepare values to predict
        values_to_predict_in_s = np.arange(start, end + granularity, granularity)

        # Last value to predict could be greater than the end
        if values_to_predict_in_s[-1] > end:
            values_to_predict_in_s[-1] = end

        return self.predict(values_to_predict_in_s)

    def plot(
        self,
        start: Optional[float] = None,
        end: Optional[float] = None,
        granularity: Optional[float] = None,
        plot_ipus: Optional[bool] = None,
        show: Optional[bool] = None,
        save_fname: Optional[str] = None,
        **kwargs,
    ):
        """
        Plot the predictions between the given
        start and end, and with the given granularity.

        Parameters
        ----------
        start: Optional[float]
            A starting point in time to predict. Default is self.start()
        end: Optional[float]
            An ending point in time to predict. Default is self.end()
        granularity: Optional[float]
            The step in time in which to predict from the time series. Default is 0.01
        plot_ipus: Optional[bool]
            Whether to plot also the InterPausalUnits feature values. Default is True.
        show: Optional[bool]
            Whether to show the plot. Default is True.
        save_fname: Optional[str]
            The fname to pass to plt.savefig(). If provided the plot will be saved.
        """
        if start is None:
            start = self.start()

        if end is None:
            end = self.end()

        if granularity is None:
            granularity = 0.01

        if plot_ipus is None:
            plot_ipus = True

        if show is None:
            show = True

        xs = np.arange(start, end + granularity, granularity)
        values_to_predict_in_s = deepcopy(xs)
        # Last value to predict could be greater than the end
        if values_to_predict_in_s[-1] > self.end():
            values_to_predict_in_s[-1] = self.end()

        ys = self.predict(values_to_predict_in_s)
        plt.plot(xs, ys, **kwargs)

        if plot_ipus:
            ipus_values = [
                ipu.feature_value(self.feature)
                for ipu in self.ipus
                if ipu.start >= start and ipu.end <= end
            ]
            ipus_starts = [
                ipu.start for ipu in self.ipus if ipu.start >= start and ipu.end <= end
            ]
            ipus_ends = [
                ipu.end for ipu in self.ipus if ipu.start >= start and ipu.end <= end
            ]
            plt.hlines(
                y=ipus_values, xmin=ipus_starts, xmax=ipus_ends, linewidth=4.4, **kwargs
            )

        plt.xlabel("Time (seconds)")
        plt.ylabel(self.feature)

        if save_fname is not None:
            plt.savefig(save_fname)

        if show:
            plt.show()

    def outlier_ipus(self):
        """
        Returns the amount of InterPausalUnits with an outlier feature value.
        """
        return self.outliers
