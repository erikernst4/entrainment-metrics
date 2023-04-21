import io
import os
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, List, Optional

import audiofile
import numpy as np
import opensmile
import pandas as pd
from allosaurus.app import read_recognizer
from parselmouth.praat import run_file
from scipy.io import wavfile


class InterPausalUnit:
    """
    It's an interval of time between silences of a speaker in a conversation.


    Attributes
    ----------
    start: float
        Start time of the IPU.

    end: float
        End time of the IPU.

    features_values: Optional[Dict[str, float]]
        A dictionary with the features calculated and their values.
    """

    def __init__(
        self,
        start: float,
        end: float,
        features_values: Optional[Dict[str, float]] = None,
    ) -> None:
        self.start = start
        self.end = end
        self.features_values = features_values if features_values is not None else {}

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

    def feature_value(
        self,
        feature: str,
    ) -> float:
        """
        Return the value for the feature given if already extracted
        """
        if self.features_values is None or feature not in self.features_values:
            raise ValueError(f"Feature {feature} not extracted yet")
        return self.features_values[feature]

    def calculate_features(
        self,
        audio_file: Path,
        pitch_gender: Optional[str] = None,
        extractor: Optional[str] = None,
    ) -> Optional[Dict[str, float]]:
        """
        Feature extraction for an InterPausalUnit.

        Parameters
        ----------
        audio_file: Path
            A path to a wav file.
        pitch_gender: Optional[str]
            Useful for a more accurate praat extraction. "M" or "F", or None.
        extractor: Optional[str]
            The extractor to calculate features. It can be either "praat" ,"opensmile", or "allosaurus"/"speech-rate". Default is "opensmile".
        Returns
        -------
        Dict[str, float]
            A dictionary with the value for each feature calculated.
        """
        available_extractors = ["praat", "opensmile", "allosaurus", "speech-rate"]
        # Set opensmile as default
        if extractor is None and self.features_values is None:
            extractor = "opensmile"

        if extractor is None:
            pass
        elif extractor not in available_extractors:
            raise ValueError('Not a valid extractor')
        elif extractor == "praat":
            self._calculate_praat_features(audio_file, pitch_gender)  # type: ignore
        elif extractor == "opensmile":
            self._calculate_opensmile_features(audio_file)  # type: ignore
        elif extractor in ["speech-rate", "allosaurus"]:
            self._calculate_speech_rate(audio_file)

        return self.features_values

    def _calculate_praat_features(
        self,
        audio_file: Path,
        pitch_gender: Optional[str],
    ) -> None:
        """
        Return the IPU values of the standard acoustics features


        This features are calculated with praat using the script
        in praat_scripts
        """
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

        audio_file = Path(audio_file)
        audio_file_absolute = os.fspath(audio_file.resolve())
        praat_script_absolute = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                'extractStandardAcoustics.praat',
            )
        )
        f = io.StringIO()
        with redirect_stdout(f):
            run_file(
                praat_script_absolute,
                audio_file_absolute,
                str(self.start),
                str(self.end),
                str(min_pitch),
                str(max_pitch),
            )

        # Parse results
        result: List[str] = f.getvalue().rstrip().splitlines()
        features_results: Dict[str, float] = {}
        for line in result:
            feature, value = line.split(":")
            if value != "--undefined--":
                features_results[feature] = float(value)

        self.features_values.update(features_results)

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
        self.features_values.update(
            self._convert_opensmile_output(opensmile_features_csv)
        )

    def _convert_opensmile_output(self, df: pd.DataFrame) -> Dict[str, float]:
        return df.to_dict(orient='records')[0]

    def _calculate_speech_rate(self, audio_file: Path, lang_id: Optional[str] = None):
        if lang_id is None:
            lang_id = "ipa"
        # Create cropped wav
        audio_file = Path(audio_file)
        audio_file_absolute = os.fspath(audio_file.resolve())
        samplerate, data = wavfile.read(audio_file)
        wav_start, wav_end = int(self.start * samplerate), int(self.end * samplerate)
        cropped_wav: np.ndarray = data[wav_start:wav_end]

        # Temporary save the cropped wav
        cropped_wav_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                'temp_crop.wav',
            )
        )
        wavfile.write(cropped_wav_path, samplerate, cropped_wav)

        # Phonemize wav
        model = read_recognizer()
        ipu_phones = model.recognize(cropped_wav_path, lang_id=lang_id)

        # Calculate speech rate
        ipu_phones_qty = len(ipu_phones.split())
        ipu_speech_rate = ipu_phones_qty / self.duration()
        self.features_values.update({"speech_rate": ipu_speech_rate})

        # Remove tempoary wav
        os.remove(cropped_wav_path)
