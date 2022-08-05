import argparse
import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.io import wavfile

arg_parser = argparse.ArgumentParser(
    description="Cut wav files for each task and create its .words files"
)
arg_parser.add_argument(
    "-s", "--session-folder", type=str, help="Folder with wavs, tasks and words"
)
arg_parser.add_argument(
    "-o", "--output-path", type=str, help="Directory to save the output files"
)


def read_words(path: Path) -> Tuple[List[str], List[str]]:
    words_A: List[str] = []
    words_B: List[str] = []
    for filename in glob.glob(os.path.join(path, "*.words")):
        file_extensions = filename.split(".")
        with open(filename, encoding="utf-8", mode="r") as word_file:
            lines: List[str] = word_file.read().splitlines()

            if "A" in file_extensions:
                words_A = lines
            elif "B" in file_extensions:
                words_B = lines

    return words_A, words_B


def read_wavs(
    path: Path,
) -> Tuple[Optional[Tuple[int, np.ndarray]], Optional[Tuple[int, np.ndarray]]]:
    wav_A = None
    wav_B = None
    for wav_file in glob.glob(os.path.join(path, "*.wav")):
        file_extensions = wav_file.split(".")

        samplerate, data = wavfile.read(wav_file)
        wav: Optional[Tuple[int, np.ndarray]] = (samplerate, data)

        if "A" in file_extensions:
            wav_A = wav
        elif "B" in file_extensions:
            wav_B = wav

    return wav_A, wav_B


def read_tasks(path: Path) -> Dict[int, Dict[str, Any]]:
    tasks: Dict[int, Dict[str, Any]] = {}
    for filename in glob.glob(os.path.join(path, "*.tasks")):
        with open(filename, encoding="utf-8", mode="r") as tasks_file:
            lines: List[str] = tasks_file.read().splitlines()

            task_id: int = 1
            for line in lines:
                task_start, task_end, task_label = line.split(" ")
                if task_label.startswith("Images"):
                    tasks[task_id] = {
                        "Start": float(task_start),
                        "End": float(task_end),
                        "Label": task_label,
                    }
                    task_id += 1

    return tasks


def read_session_name(path: Path) -> str:
    first_file = glob.glob(os.path.join(path, "*.*.*.*.*"))[0]
    filename = os.path.split(first_file)[1]
    session_name: str = filename.split(".1")[0]
    return session_name


def cut_wav_for_each_task(
    wav: Optional[Tuple[int, np.ndarray]],
    tasks: Dict[int, Dict[str, Any]],
    output_path: Path,
    session_name: str,
    speaker: str,
) -> None:
    for task_id, task in tasks.items():
        if wav is not None:
            samplerate, data = wav
        task_start, task_end = int(task["Start"]), int(task["End"])
        cutted_data: np.ndarray = data[task_start * samplerate : task_end * samplerate]

        task_wav_name: str = session_name + f".1.{task_id}" + f".{speaker}.wav"
        output_dir: str = os.path.join(output_path, task_wav_name)
        wavfile.write(output_dir, samplerate, cutted_data)

        print(
            f'Saved wav for task {task_id} from speaker {speaker}: {task["Start"]}s - {task["End"]}s'
        )


def create_words_for_each_task(
    words: List[str],
    tasks: Dict[int, Dict[str, Any]],
    output_path: Path,
    session_name: str,
    speaker: str,
) -> None:
    for task_id, task in tasks.items():
        words_task_name: str = session_name + f".1.{task_id}" + f".{speaker}.words"
        words_filename: str = os.path.join(output_path, words_task_name)
        with open(words_filename, encoding="utf-8", mode="w") as word_file:
            for line in words:
                start, end, word = line.split(" ")
                word_start, word_end = float(start), float(end)
                if (
                    word_start > task["Start"] and word_end < task["End"]
                ):  # TO-ASK: What if a word is between tasks?
                    word_start, word_end = (
                        word_start - task["Start"],
                        word_end - task["Start"],
                    )
                    word_file.write(f"{word_start} {word_end} {word}\n")
        print(f"Wrote .words for task {task_id}")


def main() -> None:

    args = arg_parser.parse_args()
    session_path: Path = Path(args.session_folder)

    words_A, words_B = read_words(session_path)

    wav_A, wav_B = read_wavs(session_path)

    tasks: Dict[int, Dict[str, Any]] = read_tasks(session_path)
    print(f'There are {len(tasks)} tasks in this session')

    output_path: Path = Path(args.output_path)
    session_name: str = read_session_name(session_path)
    cut_wav_for_each_task(wav_A, tasks, output_path, session_name, "A")
    cut_wav_for_each_task(wav_B, tasks, output_path, session_name, "B")

    create_words_for_each_task(words_A, tasks, output_path, session_name, "A")
    create_words_for_each_task(words_B, tasks, output_path, session_name, "B")


if __name__ == "__main__":
    main()
