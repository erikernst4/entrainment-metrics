import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import audiofile
import opensmile
import pandas as pd


class InterPausalUnit:
    """
    It's an interval of time between silences in a conversation

     ...

    Attributes
    ----------
    start: float
        Start time of the IPU
    end: float
        End time of the IPU

    """

    def __init__(
        self,
        start: float,
        end: float,
        features_values: Optional[Dict[str, float]] = None,
    ) -> None:
        self.start = start
        self.end = end
        self._features_values = features_values

    def __eq__(self, other):
        res = False
        if isinstance(other, InterPausalUnit):
            if self.start == other.start and self.end == other.end:
                res = True
        return res

    def __repr__(self):
        return f"InterPausalUnit(start={self.start}, end={self.end})"

    def duration(self) -> float:
        return self.end - self.start

    def calculate_features(
        self, audio_file: Path, pitch_gender: Optional[str], extractor: str
    ) -> Optional[Dict[str, float]]:
        """
        Given an audio_file calculate the features for the IPU inside
        """
        if extractor == "praat":
            self._calculate_praat_features(audio_file, pitch_gender)
        elif extractor == "opensmile":
            self._calculate_opensmile_features(audio_file)

        return self._features_values

    def _calculate_praat_features(self, audio_file: Path, pitch_gender: Optional[str]):
        """
        Return the IPU values of the standard acoustics features


        This features are calculated with praat using the script
        in praat_scripts
        """
        if self._features_values is None:
            min_pitch = None
            max_pitch = None
            if pitch_gender == "M":
                min_pitch = 50
                max_pitch = 300
            elif pitch_gender == "F":
                min_pitch = 75
                max_pitch = 500
            elif pitch_gender is None:
                min_pitch = 50
                max_pitch = 500
            else:
                raise ValueError("Not a valid pitch gender")

            audio_file_absolute = os.fspath(audio_file.resolve())
            result = subprocess.run(
                [
                    'praat',
                    './praat_scripts/extractStandardAcoustics.praat',
                    # audio_file.absolute().as_posix(),
                    audio_file_absolute,
                    str(self.start),
                    str(self.end),
                    str(min_pitch),
                    str(max_pitch),
                ],
                stdout=subprocess.PIPE,
                check=True,
            )
            # Parse results
            output_lines: List[str] = result.stdout.decode().rstrip().splitlines()
            features_results: Dict[str, float] = {}
            for line in output_lines:
                feature, value = line.split(":")
                if value != "--undefined--":
                    features_results[feature] = float(value)

            self._features_values = features_results

    def _calculate_opensmile_features(self, audio_file: Path):
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        signal, sampling_rate = audiofile.read(
            audio_file,
            offset=self.start,
            duration=self.duration(),
        )
        opensmile_features_csv = smile.process_signal(signal, sampling_rate)
        self._features_values = self._convert_opensmile_output(opensmile_features_csv)

    def _convert_opensmile_output(self, df: pd.DataFrame) -> Dict[str, float]:
        df_to_dict = df.to_dict('index')
        features_dict = next(iter(df_to_dict.items()))[1]
        return features_dict
