Time-Aligned Moving Average (TAMA)
==================================

Using Frames
------------

If you have already have the list of InterPausalUnit's inside a frame, you can create your Frame object with its constructor:

.. code-block:: python

   from speechalyze.tama import Frame, MissingFrame
   frame = Frame(
       start=8.0,
       end=24.0,
       is_missing=False,
       interpausal_units=[
           InterPausalUnit(8.0, 12.0, {'F0_MAX': 200.002}),
           InterPausalUnit(16.0, 24.0, {'F0_MAX': 300.002}),
       ]
   )

   missing_frame = MissingFrame(start=0.0, end=0.4)

In case you don't have the specific list of InterPausalUnit's for each frame inside a wav file but you do have a .word file (as described in the InterPausalUnit section), then you can get your frames like this:

.. code-block:: python

   from speechalyze.tama import get_frames
   from typing import List, Union

   some_frames: List[Union[tama.Frame, tama.MissingFrame]] = get_frames(
       wav_fname, words_fname
   )
