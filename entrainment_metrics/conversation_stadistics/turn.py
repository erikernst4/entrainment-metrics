from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ..interpausal_unit import InterPausalUnit


@dataclass
class Turn:
    start: float
    end: float
    category: str
    speaker: Optional[str] = None

    def duration(self) -> float:
        return self.end - self.start


def is_interval_inside_an_ipu(interval_start, interval_end, ipus_list):
    res = False
    for ipu in ipus_list:
        is_interval_inside_ipu = ipu.start < interval_start and ipu.end > interval_end
        if is_interval_inside_ipu:
            res = True
            break
    return res


def was_speaker_talking_at_point_in_time(point_in_time, ipus_list):
    res = False
    for ipu in ipus_list:
        if point_in_time >= ipu.start and point_in_time <= ipu.end:
            res = True
            break
    return res


def get_turns(
    turns_times_list: List[Tuple[float, float]],
    ipus_other_speaker: List[InterPausalUnit],
) -> List[Turn]:
    res = []
    for turn_start, turn_end in turns_times_list:
        # Classify turn
        is_embedded = is_interval_inside_an_ipu(
            turn_start, turn_end, ipus_other_speaker
        )
        if is_embedded:
            turn_category = "embedded"
        elif was_speaker_talking_at_point_in_time(turn_start, ipus_other_speaker):
            turn_category = "starts_with_overlap"
        else:
            turn_category = "starts_without_overlap"
        res.append(Turn(start=turn_start, end=turn_end, category=turn_category))
    return res


def has_speaker_talked_in_interval(
    speaker_ipus: List[InterPausalUnit], interval_start: float, interval_end: float
) -> bool:
    res = False
    for ipu in speaker_ipus:
        interval_inside_ipu = ipu.start < interval_start and ipu.end > interval_end
        interval_has_intersection_with_ipu = (
            ipu.start > interval_start and ipu.start < interval_end
        ) or (ipu.end > interval_start and ipu.end < interval_end)
        if interval_inside_ipu or interval_has_intersection_with_ipu:
            res = True
            break
    return res


def _collapse(
    turns_times_list: List[Tuple[float, float]],
    ipus_other_speaker: List[InterPausalUnit],
) -> List[Tuple[float, float]]:
    res = []
    if len(turns_times_list) == 1:
        return turns_times_list

    for i in range(len(turns_times_list) - 1):
        turn_1_start, turn_1_end = turns_times_list[i]
        turn_2_start, turn_2_end = turns_times_list[i + 1]
        can_collapse = not has_speaker_talked_in_interval(
            ipus_other_speaker, turn_1_end, turn_2_start
        )
        if can_collapse:
            res.append((turn_1_start, turn_2_end))
            rest = [
                turn for index, turn in enumerate(turns_times_list) if index > i + 1
            ]
            res += rest
            break
        else:
            res.append(turns_times_list[i])
            if i == len(turns_times_list) - 2:
                res.append(turns_times_list[i + 1])
    return res


def turns_from_speaker(
    ipus_speaker: List[InterPausalUnit],
    ipus_other_speaker: List[InterPausalUnit],
) -> List[Turn]:
    turns_times = [(ipu.start, ipu.end) for ipu in ipus_speaker]
    for i in range(len(turns_times)):
        turns_times = _collapse(turns_times, ipus_other_speaker)
    return get_turns(turns_times, ipus_other_speaker)


def get_turns_from_conversation(
    ipus_speaker1: List[InterPausalUnit], ipus_speaker2: List[InterPausalUnit]
) -> Tuple[List[Turn], List[Turn]]:
    turns_speaker1 = turns_from_speaker(ipus_speaker1, ipus_speaker2)
    turns_speaker2 = turns_from_speaker(ipus_speaker2, ipus_speaker1)
    return turns_speaker1, turns_speaker2


def sort_turn_start(turn):
    return turn.start


def get_turn_changes(turns1, turns2):
    conversation_turn_changes = []
    turns_a = deepcopy(turns1)
    turns_z = deepcopy(turns2)
    for turn in turns_a:
        turn.speaker = "A"
    for turn in turns_z:
        turn.speaker = "Z"

    all_turns = turns_a + turns_z
    # Discard embedded turns
    all_turns = [turn for turn in all_turns if turn.category != "embedded"]

    # Sort turns by start
    all_turns.sort(key=sort_turn_start)

    for i in range(1, len(all_turns)):
        previous_turn = all_turns[i - 1]
        actual_turn = all_turns[i]
        if previous_turn.speaker != actual_turn.speaker:
            conversation_turn_changes.append(actual_turn.start)
    return conversation_turn_changes
