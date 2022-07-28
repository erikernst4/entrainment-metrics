import argparse
from pathlib import Path

from scipy.io import wavfile

arg_parser = argparse.ArgumentParser(
    description="Generate a times series for a speaker for a task"
)
arg_parser.add_argument(
    "-a", "--audio-file", type=str, help="Audio .wav file for a speaker for a task"
)
arg_parser.add_argument(
    "-w", "--words-file", type=str, help=".words file for a speaker for a task"
)


def get_relevant_frames_utterances(words_fname):
    relevant_frames_utterances = []
    relevant_frames_utterances_duration_sum = 0
    with open(words_fname, encoding="utf-8", mode="r") as word_file:
        rfu_started = False
        rfu_start = 0.0
        last_end = 0.0
        while line := word_file.readline().rstrip():  # Efficient reading
            start, end, word = line.split()
            if not rfu_started and word == "#":
                rfu_start = 0.0
                last_end = 0.0
            elif not rfu_started and word != "#":
                rfu_start = start
                last_end = end
                rfu_started = True
            elif rfu_started and word != "#":
                last_end = end
            elif rfu_started and word == "#":
                rfu_start, rfu_end = float(rfu_start), float(last_end)
                relevant_frames_utterances.append((rfu_start, rfu_end))
                relevant_frames_utterances_duration_sum += rfu_end - rfu_start
                rfu_started = False
        if rfu_start and last_end:  # Last RFU if existent
            relevant_frames_utterances.append((float(rfu_start), float(last_end)))

    return relevant_frames_utterances, relevant_frames_utterances_duration_sum


def is_frame_in_relevant_frame_utterance(
    relevant_frames_utterances, frame_start, frame_end, samplerate
):
    # POSSIBLE TO-DO: make a logorithmic search
    # Convert frame ends to seconds
    frame_start_in_s = frame_start / samplerate
    frame_end_in_s = frame_end / samplerate
    for rfu_start, rfu_end in relevant_frames_utterances:
        max_start = max(rfu_start, frame_start_in_s)
        min_end = min(rfu_end, frame_end_in_s)
        if max_start < min_end:
            return True
    return False


def separate_frames(relevant_frames_utterances, data, samplerate):
    FRAME_LENGHT = 16 * samplerate
    TIME_STEP = 8 * samplerate

    frames = []
    audio_length = data.shape[0]

    frame_start, frame_end = 0, FRAME_LENGHT
    while frame_start < audio_length:
        if is_frame_in_relevant_frame_utterance(
            relevant_frames_utterances, frame_start, frame_end, samplerate
        ):
            frames.append((frame_start, frame_end))

        frame_start += TIME_STEP
        frame_end += TIME_STEP
        if frame_end > audio_length:
            frame_end = audio_length - 1

    return frames


def calculate_feature():
    pass


def main() -> None:
    args = arg_parser.parse_args()

    wav_fname = Path(args.audio_file)
    samplerate, data = wavfile.read(wav_fname)
    print(f"Samplerate: {samplerate}")
    print(f"Audio data shape: {data.shape}")
    print(f"Audio data dtype: {data.dtype}")
    print(f"min, max: {data.min()}, {data.max()}")
    print(f"Lenght: {data.shape[0]/samplerate} s")

    words_fname = Path(args.words_file)
    (
        relevant_frames_utterances,
        relevant_frames_utterances_duration_sum,
    ) = get_relevant_frames_utterances(words_fname)
    print(f"Amount of RFUs: {len(relevant_frames_utterances)}")
    print(f"Duration sum of RFUs: {relevant_frames_utterances_duration_sum}s")

    frames = separate_frames(relevant_frames_utterances, data, samplerate)
    print(f"Amount of frames: {len(frames)}")
    print(frames)

    # time_series = calculate_feature()


if __name__ == "__main__":
    main()
