from typing import List, Optional, Tuple, Dict
import numpy as np
from scipy.io import wavfile
from pathlib import Path
from interpausal_unit import InterPausalUnit
class APEvolutionFunction:
    """
    

     ...

    Attributes
    ----------
    granularity: float
        The lenght of the partitions to calculate knn 
    end: float
        End time of the audio. The last point in time in the domain.
    feature: str

    extractor: Optional[str]

    k: int

    granularity: float

    interpausal_units: List[InterPausalUnit]

    wav_file: str

    pitch_gender: Optional[str]

    """
    def __init__(
        self,
        feature: str,
        extractor: Optional[str],
        k: int,
        granularity: float,
        interpausal_units: List[InterPausalUnit],
        wav_file: str,
        pitch_gender: Optional[str],
    ) -> None:
        if len(interpausal_units) < k:
            raise ValueError(
                "k cannot be bigger than the amount of interpausal units"
            )
        if not pitch_gender:
            pitch_gender = None
        if not extractor:
            extractor = None
        self.feature: str = feature
        self.extractor: Optional[str] = extractor
        self.pitch_gender: Optional[str] = pitch_gender
        self.k: float = k
        self.granularity: float = granularity
        self.ipus: List[InterPausalUnit]= interpausal_units
        self.wav_file: Path = Path(wav_file)
        self.results: Optional[Dict[Tuple[float, float], float]] = None

    def at(
        self,
        x: float
    ) -> float:
        res = None
        if self.results is None:
            self.calculate_knn_time_series()

        for (interval_start, interval_end), value in self.results.items():
            # TODO make logarithmic search
            if interval_start <= x and x < interval_end:
                res = value
        return value
        
    def end(
        self,
    ) -> float:

        _, data = wavfile.read(self.wav_file)
        audio_length: int = data.shape[0]
        return audio_length

    def as_list(
        self,
        start: Optional[float],
        end: Optional[float],
    ) -> Tuple[List[float], List[float]]:
        if self.results is None:
            self.calculate_knn_time_series()

        xs = []
        ys = []
        for (interval_start, interval_end), value in self.results.items():
            xs.append((interval_end + interval_start) / 2)
            ys.append(value)
        return xs, ys

    def _get_interpausal_units_middle_points_in_time(
        self,
    ) -> List[float]:
        """
        Given a list of IPUs returns a list with the middle point in time of each IPU.
        """
        ipus_middle_point_in_time: List[float] = []
        for ipu in self.ipus:
            ipu_middle_point_in_time = (ipu.start + ipu.end) / 2
            ipus_middle_point_in_time.append(ipu_middle_point_in_time)
        return ipus_middle_point_in_time

    def _get_interpausal_units_feature_values(
        self,
    ) -> List[float]:
        """
        Calculates the feature value for each IPU in interpausal_units
        """
        ipus_feature_values: List[float] = []
        for ipu in self.ipus:
            ipu_feature_value = ipu.calculate_features(self.wav_file, self.pitch_gender, self.extractor)[self.feature]  # type: ignore
            ipus_feature_values.append(ipu_feature_value)

        return ipus_feature_values

    def _ys_to_dict(
        self,
        ys: List[float],
    ) -> Dict[Tuple[float, float], float]:
        res = {}
        interval_start: float = 0.0
        for y in ys:
            res[(interval_start, interval_start + self.granularity)] = y



    def calculate_knn_time_series(
        self,
    ) -> None:
        """
        Generate an estimation of speakers’ a/p evolution functions by fitting a knn regression model.
        k: number of neighboors.


        O(n) being n the amount of interpausal units
        """

        time_series: List[float] = []

        ipus_middle_points_in_time = self._get_interpausal_units_middle_points_in_time()

        ipus_feature_values = self._get_interpausal_units_feature_values

        if self.k == 1:
            self.results = self._ys_to_dict(ipus_feature_values)

        # Initialize neighboors
        k_nearest_neighboors: List[float] = ipus_feature_values[:k]
        first_neighboor_index: int = 0
        time_series.append(np.mean(k_nearest_neighboors))  # type: ignore

        for ipu_index, ipu_middle_point_in_time in enumerate(
            ipus_middle_point_in_time[1:], 1
        ):
            # Calculate k nearest neighboors
            next_neighboor_index = first_neighboor_index + k
            if ipu_index >= next_neighboor_index:
                # If current ipu falls outside of the k_nearest_neighboors, recalculate
                (
                    first_neighboor_index,
                    k_nearest_neighboors,
                ) = calculate_k_nearest_neighboors_from_index(
                    ipu_index, k, ipus_middle_point_in_time, ipus_feature_values
                )
            elif next_neighboor_index < len(ipus_middle_point_in_time):
                # EXPLAIN
                ipu_distance_with_the_first_neighboor = (
                    ipu_middle_point_in_time
                    - ipus_middle_point_in_time[first_neighboor_index]
                )
                ipu_distance_with_the_next_neighboor = (
                    ipus_middle_point_in_time[next_neighboor_index]
                    - ipu_middle_point_in_time
                )
                if (
                    ipu_distance_with_the_next_neighboor
                    < ipu_distance_with_the_first_neighboor
                ):
                    # Remove first and append next neighboor
                    k_nearest_neighboors = k_nearest_neighboors[1:]
                    k_nearest_neighboors.append(ipus_feature_values[next_neighboor_index])
                    first_neighboor_index += 1

            time_series.append(np.mean(k_nearest_neighboors))  # type: ignore

        return time_series
