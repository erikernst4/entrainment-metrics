from pathlib import Path
from typing import List, Union

from frame import Frame, MissingFrame


def calculate_time_series(
    feature: str, frames: List[Union[Frame, MissingFrame]], audio_file: Path
) -> List[float]:
    """
    Generate a time series of the frames values for the feature given
    """
    time_series: List[float] = []
    for frame in frames:
        frame_time_series_value = frame.calculate_feature_value(feature, audio_file)
        time_series.append(frame_time_series_value)
    return time_series
