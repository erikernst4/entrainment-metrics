Time-Aligned Moving Average (TAMA)
==================================

Using Frames
------------

If you have already have the list of InterPausalUnit's inside a frame, you can create your Frame object with its constructor:

.. code-block:: python

   from entrainment_metrics.tama import Frame, MissingFrame
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

   from entrainment_metrics.tama import get_frames
   from typing import List, Union

   some_frames: List[Union[tama.Frame, tama.MissingFrame]] = get_frames(
       wav_fname, words_fname
   )

Constructing time series of acoustic-prosodic features
------------------------------------------------------

Once you have a list of frames corresponding to the frames in a audio you can generate time-series data of several a/p features. If the InterPausalUnit's inside of your frames already have the features of interest calculated then you can directly calculate the time series like this:

.. code-block:: python

   from entrainment_metrics.tama import calculate_time_series
   from typing import List

   time_series_a: List[float] = calculate_time_series(
       args.feature, frames_a,
       feature="FEATURE_CALCULATED",
       frames=some_frames,
   )

But, if your IPUs don't have their feature values calculated then you can the known entrainment_metrics extractors:

.. code-block:: python

   from entrainment_metrics.tama import calculate_time_series
   from typing import List

   time_series_a: List[float] = calculate_time_series(
       args.feature, frames_a,
       feature="F0_MAX",
       frames=some_frames,
       audio_file="path/to/audio.wav",
       extractor="praat",
       pitch_gender="F",
   )

Sample cross-correlation as a proxy for entrainment
---------------------------------------------------

The sample cross-correlation is a measure which aims at capturing the correlation between two series as one of them is lagged (i.e., its points are shifted a number of positions). Intuitively, it can be interpreted similarly to Pearson’s correlation coefficient between a time-series and a lagged version of another one.

Having two time series calculated you can calculate the sample cross-correlation as simply as this:

.. code-block:: python

   from entrainment_metrics.tama import calculate_sample_correlation
   from typing import List

   sample_cross_correlations: List[float] = calculate_sample_correlation(
       time_series_a=time_series_a,
       time_series_b=time_series_b,
       lags=an_amount_of_lags,
   )

Measuring acoustic-prosodic synchrony
-------------------------------------

We provide two ways of measuring acoustic-prosodic synchrony in the library. Signed and Unsigned Synchrony Measure.

For Signed Synchrony Measure positive values of S represent positive synchrony (or entrainment) in a straightforward way, and negative values represent negative synchrony (disentrainment).

In the other hand, for Unsigned Synchrony Measure, by taking the absolute value, this second measure gives equal treatment to positive and negative synchrony values. In other words, high values of the time series are indicative of high levels of either entrainment or disentrainment; and low values correspond to a total lack of coordination in either direction.

Here's an example of how to get both metrics:

.. code-block:: python

   from entrainment_metrics.tama import signed_synchrony, unsigned_synchrony
   res_signed_synchrony = signed_synchrony(
       time_series_a=time_series_a,
       time_series_b=time_series_b,
       lags=an_amount_of_lags,
   )

   res_unsigned_synchrony = unsigned_synchrony(
       time_series_a=time_series_a,
       time_series_b=time_series_b,
       lags=an_amount_of_lags,
   )

