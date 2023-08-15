# entrainment-metrics
[![Documentation Status](https://readthedocs.org/projects/entrainment-metrics/badge/?version=latest)](https://entrainment-metrics.readthedocs.io/en/latest/?badge=latest)

entrainment-metrics is all about being able to measure entrainment. Entrainment in spoken dialogue is commonly defined as a tendency of a speaker to adapt some properties of her speech to match her interlocutor’s. With this library you’ll be able to measure entrainment along one dimension: acoustic-prosodic (a/p) features.

Checkout [the docs](https://entrainment-metrics.readthedocs.io/en/latest/) and the [Getting started](https://entrainment-metrics.readthedocs.io/en/latest/usage/getting_started.html#getting-started) page for a deeper dive into the library!

## Installation
- To use entrainment_metrics, first install it using pip:

```bash
pip install entrainment_metrics
```
- Speech feature extraction
If you'll be using praat for feature extraction it's also required the command-line tool ffmpeg to be installed on your system, which is available from most package managers:

```bash
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
```
And for installing praat on Ubuntu or Debian:

```bash
sudo apt update && sudo apt install praat
```
