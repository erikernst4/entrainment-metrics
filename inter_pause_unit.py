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
    """

    def __init__(
        self,
        start: float,
        end: float,
    ) -> None:
        self.start = start
        self.end = end
