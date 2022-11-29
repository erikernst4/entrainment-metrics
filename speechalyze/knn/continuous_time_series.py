import warnings
from copy import deepcopy
from typing import List, Optional

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from speechalyze import InterPausalUnit


class TimeSeries:
    """
    TODO


     ...

    Attributes
    ----------
    feature: str
        The feature to get the value from each InterPausalUnit

    interpausal_units: List[InterPausalUnit]
        An ordered list of InterPausalUnit's

    method: str
        The method to be used to predict

    k: Optional[int]
        The amount of neighbors to use in KNeighborsRegressor

    MAX_DEVIATIONS: int
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
        self.ipus: List[InterPausalUnit] = deepcopy(interpausal_units)

        self.feature = feature

        self.ipus_feature_values = self._get_interpausal_units_feature_values()

        # Removes IPUs with an outlier feature value and their values in ipus_feature_values
        self._prepare_data(MAX_DEVIATIONS)

        if method == "knn":
            if k is not None and len(interpausal_units) < k:
                raise ValueError(
                    "k cannot be bigger than the amount of interpausal units"
                )

            self.model = KNeighborsRegressor(n_neighbors=k, **kwargs)

            # Define X without outliers IPUs
            X = self._get_middle_points_in_time()
            X = X.reshape(-1, 1)

            self.model.fit(X, self.ipus_feature_values)
        else:
            # Here is some space to build your own model!
            raise ValueError("Model to be implemented")

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
        return self.ipus[0].start

    def end(
        self,
    ) -> float:
        return self.ipus[-1].end

    def predict(
        self,
        X,
    ) -> np.ndarray:
        # Convert float to expected predict type
        if isinstance(X, float):
            X = np.array([X]).reshape(-1, 1)
        for x in X:
            if x > self.end():
                warnings.warn(
                    f"""Out of bounds {x}: A value in X is greater than TimeSeries end.
                    Remember the end of a TimeSeries is the end of the last non-outlier IPU.
                """
                )
            if x < self.start():
                warnings.warn(
                    f"""Out of bounds {x}: A value in X is smaller than TimeSeries start.
                    Remember the start of a TimeSeries is the start of the first non-outlier IPU.
                """
                )
        return self.model.predict(X)
