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

Constructing time series of acoustic-prosodic features
------------------------------------------------------

Once you have a list of frames corresponding to the frames in a audio you can generate time-series data of several a/p features. If the InterPausalUnit's inside of your frames already have the features of interest calculated then you can directly calculate the time series like this:

.. code-block:: python

   from speechalyze.tama import calculate_time_series
   from typing import List

   time_series_a: List[float] = calculate_time_series(
       args.feature, frames_a,
       feature="FEATURE_CALCULATED",
       frames=some_frames,
   )

But, if your IPUs don't have their feature values calculated then you can the known speechalyze extractors:

.. code-block:: python

   from speechalyze.tama import calculate_time_series
   from typing import List

   time_series_a: List[float] = calculate_time_series(
       args.feature, frames_a,
       feature="F0_MAX",
       frames=some_frames,
       audio_file="path/to/audio.wav",
       extractor="praat",
       pitch_gender="F",
   )
