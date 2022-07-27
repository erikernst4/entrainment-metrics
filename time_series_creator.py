import os
import sys
import glob
from pathlib import Path
import argparse
from scipy.io import wavfile

arg_parser = argparse.ArgumentParser(description="Generate a times series for a speaker for a task")
arg_parser.add_argument("-a", "--audio-file", type=str, help="Audio .wav file for a speaker for a task")
arg_parser.add_argument("-w", "--words-file", type=str, help=".words file for a speaker for a task")

def get_relevant_frames_utterances(words_fname):
    relevant_frames_utterances = []
    with open(words_fname, encoding="utf-8", mode="r") as word_file:
        rfu_started = False
        rft_start = 0.0
        last_end = 0.0
        while (line := word_file.readline().rstrip()):
            start, end, word = line.split()
            if not rfu_started and word == "#":
                rft_start = 0.0
                last_end = 0.0
            elif not rfu_started and word != "#":
                rfu_start = start
                last_end = end
                rfu_started = True
            elif rfu_started and word != "#":
                last_end = end
            elif rfu_started and word == "#":
                relevant_frames_utterances.append((rfu_start, last_end))
                rfu_started = False
        if rfu_start and last_end: # Last RFU if existent
            relevant_frames_utterances.append((rfu_start, last_end))

    return relevant_frames_utterances

def separate_frames(relevant_frames_utterances, data, samplerate):
    pass

def calculate_feature(frames):
    pass

def main() -> None:
    args = arg_parser.parse_args()

    wav_fname = Path(args.audio_file)
    samplerate, data = wavfile.read(wav_fname)

    words_fname = Path(args.words_file)
    relevant_frames_utterances = get_relevant_frames_utterances(words_fname)
    print(relevant_frames_utterances)

    frames = separate_frames(relevant_frames_utterances, data, samplerate)

    time_series = calculate_feature(frames)

if __name__ == "__main__":
    main()
