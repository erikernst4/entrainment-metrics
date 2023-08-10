Installation
------------

To use entrainment_metrics, first install it using pip:

.. code-block:: console

   pip install entrainment-metrics

It's also required the command-line tool ffmpeg to be installed on your system, which is available from most package managers:

.. code-block:: console

   # on Ubuntu or Debian
   sudo apt update && sudo apt install ffmpeg

   # on Arch Linux
   sudo pacman -S ffmpeg

   # on MacOS using Homebrew (https://brew.sh/)
   brew install ffmpeg

   # on Windows using Chocolatey (https://chocolatey.org/)
   choco install ffmpeg

   # on Windows using Scoop (https://scoop.sh/)
   scoop install ffmpeg

For speech feature extraction it is also required praat:

.. code-block:: console

   # on Ubuntu or Debian
   sudo apt update && sudo apt install praat


Troubleshoot
^^^^^^^^^^^^

- cannot import name 'quote' from 'urllib' (/usr/lib/python3.8/urllib/__init__.py)

    This error comes from a problem between parselmouth and praat-parselmouth. If you bump into this error while installing the library, you can solve it by running the following:
    
    .. code-block:: console

       pip uninstall parselmouth && pip install praat-parselmouth

- Please, if you have any other error feel free to leave an issue in the github repository.
