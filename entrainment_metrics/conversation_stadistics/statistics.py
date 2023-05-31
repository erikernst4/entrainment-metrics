from typing import Any, Dict, List

import numpy as np

from ..interpausal_unit import InterPausalUnit
from .turn import get_turn_from_conversation


def add_turn_statistics_to_analysis(
    analysis: Dict[str, Any],
    ipus_speaker1: List[InterPausalUnit],
    ipus_speaker2: List[InterPausalUnit],
    speaker1_name: str,
    speaker2_name: str,
) -> None:
    turns_speaker1, turns_speaker2 = get_turn_from_conversation(
        ipus_speaker1, ipus_speaker2
    )

    speakers_turns = {speaker1_name: turns_speaker1, speaker2_name: turns_speaker2}

    for speaker_name, turns_list in speakers_turns.items():
        analysis[conversation][f"{speaker_name}_turns"] = len(turns_list)
        analysis[conversation][f"{speaker_name}_turns_that_start_with_overlap"] = len(
            [turn for turn in turns_list if turn.category == "starts_with_overlap"]
        )
        analysis[conversation][
            f"{speaker_name}_turns_that_start_without_overlap"
        ] = len(
            [turn for turn in turns_list if turn.category == "starts_without_overlap"]
        )
        analysis[conversation][f"{speaker_name}_embedded_turns"] = len(
            [turn for turn in turns_list if turn.category == "embedded"]
        )
        analysis[conversation][f"{speaker_name}_turn_mean_duration"] = np.mean(
            [turn.duration() for turn in turns_list]
        )

    both_turn_list = turns_speaker1 + turns_speaker2
    analysis[conversation][f"turn_mean_duration"] = np.mean(
        [turn.duration() for turn in both_turn_list]
    )

    turn_changes = get_turn_changes(turns_speaker1, turns_speaker2)
    analysis[conversation]["turn_changes"] = len(turn_changes)
