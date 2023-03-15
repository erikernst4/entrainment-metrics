.. _getting_started:

Getting started
===============

In order to use this library you'll need at least two things: a wav file for each speaker and the start and end of each InterPausalUnit. What is an Interpausal Unit? It's an interval of time between silences of a single speaker in a conversation. You only need the start and end of each Interpausal Unit. What if you don't have that piece of information? Well, the task you need to solve is Voice Activity Detection (VAD), there are many good tools you can use to solve this!

Creating Interpausal Units
--------------------------

If you already have the Interpausal Units from a wav file, you can create your InterPausalUnit objects like this (note that feature_values are an optional parameter):

.. code-block:: python

   from entrainment_metrics import InterPausalUnit

   InterPausalUnit(
      start=0.0,
      end=4.0,
      feature_values={'F0_MAX': 100.003}
   )

If you only have the start and end from each InterPausalUnit you can use opensmile or the praat script from this library to calculate the features available with each extractor.
For example, if you have a list of InterPausalUnit's called "ipus":


.. code-block:: python

   for ipu in ipus:
        ipu.calculate_features(
            audio_file="path/to/file.wav",
            pitch_gender="F",
            extractor="praat",
        )


In case you have a .words file that follows the format '{start_time} {end_time} {word}' for each line (where start_time and end_time are floats and word is a string with "#" reserved for silences), then you can use the following method to get your IPUs:

.. code-block:: python

    from entrainment_metrics import get_interpausal_units,InterPausalUnit
    from typing import List

    ipus: List[InterPausalUnit] = get_interpausal_units(words_fname)

For further information check the :ref:`ipu` documentation.

Approximating the evolution of each speaker’s a/p features
----------------------------------------------------------

Once you have your InterPausalUnits, the next step towards measuring entrainment is to approximate the evolution of each speaker’s a/p features. With this library you can follow two paths:

- Discrete approximation -> :ref:`tama`.
- Continuous approximation -> :ref:`continuous_time_series`

For in-depth information for choosing one, please refer to the referenced papers. For beginners, we recommend you to go straight to the Continuous TimeSeries.

In the next sections you'll learn how to follow each path and get your entrainment metrics.
