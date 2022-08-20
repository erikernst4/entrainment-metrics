from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from interpausal_unit import InterPausalUnit


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
    interpausal_units: List[InterPausalUnit]
        The IPUs that fall inside of the frame
    """

    def __init__(
        self,
        start: float,
        end: float,
        is_missing: bool,
        interpausal_units: Optional[List[InterPausalUnit]],
    ) -> None:
        self.start = start
        self.end = end
        self.is_missing = is_missing
        if interpausal_units is not None:
            self.interpausal_units = interpausal_units

    def calculate_feature_value(
        self, feature: str, audio_file: Path, pitch_gender: str
    ) -> float:
        """
        Return the frame's value for the feature given

        ----
        This value is calculated as the duration-weighted mean
        of the value for the feature of each IPU inside the frame

        Cite Interspeech2016
        """

        IPUs_duration_weighten_mean_values: List[float] = []
        IPUs_duration_sum = self.calculate_IPUs_duration_sum()

        for interpausal_unit in self.interpausal_units:
            IPU_features_results: Dict[
                str, float
            ] = interpausal_unit.calculate_features(
                audio_file,
                pitch_gender,
            )
            IPU_feature_value: float = IPU_features_results[feature]
            IPU_duration_weighten_mean_value: float = (
                IPU_feature_value * interpausal_unit.duration()
            ) / IPUs_duration_sum
            IPUs_duration_weighten_mean_values.append(IPU_duration_weighten_mean_value)

        return sum(IPUs_duration_weighten_mean_values)

    def calculate_IPUs_duration_sum(self) -> float:
        res: float = 0.0
        for interpausal_unit in self.interpausal_units:
            res += interpausal_unit.duration()
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
            interpausal_units=None,
        )

    def calculate_feature_value(
        self,
        feature: str,
        audio_file: Path,
        pitch_gender: str,  # pylint: disable=unused-argument
    ) -> float:
        return np.nan
