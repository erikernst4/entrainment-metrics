from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from interpause_unit import InterPauseUnit


class Frame:
    """
    An interval of time inside an audio

     ...

    Attributes
    ----------
    start: float
         Start time of the frame.
    end: float
         End time of the frame.
    is_missing: bool
        Whether the frame has no IPUs inside. In other words, if the frame is fulled with silence.
    interpause_units: List[InterPauseUnit]
        The IPUs that fall inside of the frame

    Methods
    ----------
    calculate_feature_value(feature, audio_file)
        Return the frame's value for feature given

    """

    def __init__(
        self,
        start: float,
        end: float,
        is_missing: bool,
        interpause_units: Optional[List[InterPauseUnit]],
    ) -> None:
        self.start = start
        self.end = end
        self.is_missing = is_missing
        if interpause_units is not None:
            self.interpause_units = interpause_units

    def calculate_feature_value(self, feature: str, audio_file: Path) -> float:
        """
        Return the frame's value for feature given

        ----
        This value is calculated as the duration-weighted mean
        of the value for the feature of each IPU inside the frame

        Cite Interspeech2016
        """

        IPUs_duration_weighten_mean_values: List[float] = []
        IPUs_duration_sum = self.calculate_IPUs_duration_sum()

        for interpause_unit in self.interpause_units:
            IPU_features_results: Dict[str, float] = interpause_unit.calculate_features(
                audio_file
            )
            IPU_feature_value: float = IPU_features_results[feature]
            IPU_duration_weighten_mean_value: float = (
                IPU_feature_value * interpause_unit.duration()
            ) / IPUs_duration_sum
            IPUs_duration_weighten_mean_values.append(IPU_duration_weighten_mean_value)

        return sum(IPUs_duration_weighten_mean_values)

    def calculate_IPUs_duration_sum(self) -> float:
        res: float = 0.0
        for interpause_unit in self.interpause_units:
            res += interpause_unit.duration()
        return res


class MissingFrame(Frame):
    def __init__(
        self,
        start: float,
        end: float,
    ) -> None:
        super().__init__(
            start=start,
            end=end,
            is_missing=True,
            interpause_units=None,
        )

    def calculate_time_series_value(
        self, audio_file: Path  # pylint: disable=unused-argument
    ) -> float:
        return np.nan
