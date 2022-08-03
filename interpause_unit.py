import subprocess
from pathlib import Path
from typing import Dict, List


class InterPauseUnit:
    """
    It's an interval of time between silences in a conversation

     ...

    Attributes
    ----------
    start: float
        Start time of the IPU
    end: float
        End time of the IPU


    Methods
    ----------
    duration()
        Time duration of the IPU
    calculate_features(audio_file)
        IPU values of the standard acoustics features

    """

    def __init__(
        self,
        start: float,
        end: float,
    ) -> None:
        self.start = start
        self.end = end

    def duration(self) -> float:
        return self.end - self.start

    def calculate_features(self, audio_file: Path) -> Dict[str, float]:
        """
        Return the IPU values of the standard acoustics features

        ----
        This features are calculated with praat using the script
        in praat_scripts
        """
        result = subprocess.run(
            [
                'praat',
                './praat_scripts/extractStandardAcoustics.praat',
                audio_file,
                str(self.start),
                str(self.end),
                '75',
                '500',
            ],
            stdout=subprocess.PIPE,
            check=True,
        )
        output_lines: List[str] = result.stdout.decode().rstrip().splitlines()
        features_results: Dict[str, float] = {}
        for line in output_lines:
            feature, value = line.split(":")
            features_results[feature] = float(value)
        # print(f"Feature results for IPU from {IPU_start} to {IPU_end}")
        # print(features_results)
        return features_results
