import os
import sys
import glob
from pathlib import Path
import argparse
from scipy.io import wavfile

arg_parser = argparse.ArgumentParser(description="Cut wav files for each task and create its .words files")
arg_parser.add_argument("-s", "--session-folder", type=str, help="Folder with wavs, tasks and words")
arg_parser.add_argument("-o", "--output-path", help="Directory to save the output files")

def read_wavs(path):
    wav_A = (None, None)
    wav_B = (None, None)
    for wav_file in glob.glob(os.path.join(path, "*.wav")):
        file_extensions = wav_file.split(".")

        samplerate, data = wavfile.read(wav_file)
        wav = (samplerate, data)

        if "A" in file_extensions:
            wav_A = wav
        elif "B" in file_extensions:
            wav_B = wav
                
    return wav_A, wav_A 

def read_words(path):
    words_A = []
    words_B = []
    for filename in glob.glob(os.path.join(path, "*.words")):
        file_extensions = filename.split(".")
        with open(filename, encoding="utf-8", mode="r") as word_file:
            lines = word_file.read().splitlines() 

            if "A" in file_extensions:
                words_A = lines
            elif "B" in file_extensions:
                words_B = lines
                
    return words_A, words_B




def main() -> None:

    args = arg_parser.parse_args()
    session_dir = Path(args.session_folder)

    words_A, words_B = read_words(session_dir)

    wav_A, wav_B = read_wavs(session_dir)

    print(wav_A)
    print(wav_B)

if __name__ == "__main__":
    main()
