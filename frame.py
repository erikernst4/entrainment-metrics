from typing import List, Optional

from inter_pause_unit import InterPauseUnit


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

    Methods
    ----------
    """

    def __init__(
        self,
        start: float,
        end: float,
        is_missing: bool,
        inter_pause_units: Optional[List[InterPauseUnit]],
    ) -> None:
        self.start = start
        self.end = end
        self.is_missing = is_missing
        if inter_pause_units is not None:
            self.IPUs = inter_pause_units


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
            inter_pause_units=None,
        )
