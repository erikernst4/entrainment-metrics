from typing import List, Union

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
            y = self._get_interpausal_units_feature_values(feature)
            self.model.fit(X, y)
        else:
            # Here is some space to build your own model!
            raise ValueError("Model to be implemented")

    def predict(
        self,
        X: Union[List[float], np.ndarray],
    ) -> np.ndarray:
        return self.model.predict(X)

    def _get_interpausal_units_feature_values(
        self,
        feature: str,
    ) -> List[float]:
        """
        Returns a list with the feature value for each IPU.
        """
        return [ipu.feature_value(feature) for ipu in self.ipus]

    def _get_middle_points_in_time(
        self,
    ) -> List[float]:
        """
        Returns a list with the middle point in time of each IPU.
        """
        return [(ipu.start + ipu.end) / 2 for ipu in self.ipus]
