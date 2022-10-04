from typing import List

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

from interpausal_unit import InterPausalUnit


class TimeSeries:
    """
    TODO


     ...

    Attributes
    ----------
    feature: str

    interpausal_units: List[InterPausalUnit]

    k: int

    """

    def __init__(
        self,
        feature: str,
        interpausal_units: List[InterPausalUnit],
        method: str,
        k: int = None,
        **kwargs,
    ) -> None:
        self.ipus: List[InterPausalUnit] = interpausal_units

        if method == "knn":
            if k is not None and len(interpausal_units) < k:
                raise ValueError(
                    "k cannot be bigger than the amount of interpausal units"
                )

            self.model = KNeighborsRegressor(n_neighbors=k, **kwargs)

            X = self._get_middle_points_in_time()
            X = X.reshape(-1, 1)
            y = self._get_interpausal_units_feature_values(feature)
            self.model.fit(X, y)
        else:
            # Here is some space to build your own model!
            raise ValueError("Model to be implemented")

    def predict(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        return self.model.predict(X)

    def _get_interpausal_units_feature_values(
        self,
        feature: str,
    ) -> np.ndarray:
        """
        Returns a list with the feature value for each IPU.
        """
        return np.array([ipu.feature_value(feature) for ipu in self.ipus])

    def _get_middle_points_in_time(
        self,
    ) -> np.ndarray:
        """
        Returns a list with the middle point in time of each IPU.
        """
        return np.array([(ipu.start + ipu.end) / 2 for ipu in self.ipus])
