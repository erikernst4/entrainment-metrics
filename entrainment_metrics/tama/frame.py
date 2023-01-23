from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from entrainment_metrics import InterPausalUnit


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

    def __eq__(self, other):
        res = False
        if isinstance(other, Frame):
            if (
                self.start == other.start
                and self.end == other.end
                and self.is_missing == other.is_missing
                and self.interpausal_units == other.interpausal_units
            ):
                res = True
        return res

    def __repr__(self):
        return f"Frame(start={self.start}, end={self.end}, is_missing={self.is_missing}, interpausal_units={self.interpausal_units})"

    def calculate_feature_value(
        self,
        feature: str,
        audio_file: Optional[Path] = None,
        pitch_gender: Optional[str] = None,
        extractor: Optional[str] = None,
    ) -> float:
        """
        Return the frame's value for the feature given


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
                audio_file, pitch_gender, extractor
            )  # type: ignore
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

    def __eq__(self, other):
        res = False
        if isinstance(other, MissingFrame):
            if self.start == other.start and self.end == other.end:
                res = True
        return res

    def __repr__(self):
        return f"MissingFrame(start={self.start}, end={self.end})"

    def calculate_feature_value(
        self,
        feature: str,
        audio_file: Optional[Path] = None,
        pitch_gender: Optional[str] = None,  # pylint: disable=unused-argument
        extractor: Optional[str] = None,
    ) -> float:
        return np.nan
