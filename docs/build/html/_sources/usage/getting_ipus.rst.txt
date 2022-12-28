Getting Interpausal Units
=========================

If you already have the Interpausal Units from a wav file, you can create your InterpausalUnit objects like this (note that feature_values are an optional parameter):

.. code-block:: python

   from speechalyze import InterpausalUnit

   InterPausalUnit(
      start=0.0,
      end=4.0,
      feature_values={'F0_MAX': 100.003}
   )

If you only have the start and end from each InterPausalUnit you can use opensmile or the praat script from this library to calculate the features available with each extractor.
For example, if you have a list of InterpausalUnit's called "ipus":


.. code-block:: python

   for ipu in ipus:
        ipu.calculate_features(
            audio_file="path/to/file.wav",
            pitch_gender="F",
            extractor="praat",
        )


In case you have a .word file that follows the format '{starting_time} {ending_time} {word}' for each line (where starting_time and ending_time are floats and word is a string with "#" reserved for silences), then you can use the following method to get your IPUs:

.. code-block:: python

    from speechalyze import get_interpausal_units

    ipus: List[InterPausalUnit] = get_interpausal_units(words_fname)

For further information check the InterPausalUnit documentation.

