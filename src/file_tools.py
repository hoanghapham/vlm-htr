from string import punctuation
from pathlib import Path
import os
import sys
import json
import unicodedata
import pandas as pd


def list_files(input_path: Path | str, extensions) -> list[Path]:
    if not isinstance(input_path, Path):
        input_path = Path(input_path)

    files = [
        file for file in sorted(list(input_path.glob("**/*"))) 
        if file.suffix in extensions
    ]
        
    # data = list(zip(
    #     [file.parent for file in files],
    #     [file.name for file in files]
    # ))

    return files

def check_file_stats(input_path: Path | str):
    

    if not isinstance(input_path, Path):
        input_path = Path(input_path)

    ignored_chars = punctuation + " "

    files = sorted([file for file in sorted(list(input_path.glob("**/*"))) if file.suffix == ".txt"])

    file_chars = []
    file_words = []

    for file in files:
        chars = 0
        words = 0
        
        with open(file, "r", encoding="utf-8") as f:
            for line in f.readlines():
                chars += len([c for c in line.strip() if c not in ignored_chars])
                words += len([w for w in line.strip().split(" ") if w not in ignored_chars])
        
        file_chars.append(chars)
        file_words.append(words)

    data = pd.DataFrame({
        "parent_path": [file.parent for file in files],
        "file_name": [file.stem for file in files],
        "file_chars": file_chars,
        "file_words": file_words
    })
    return data


class suppress_stdout_stderr(object):
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')

        self.old_stdout_fileno_undup    = sys.stdout.fileno()
        self.old_stderr_fileno_undup    = sys.stderr.fileno()

        self.old_stdout_fileno = os.dup ( sys.stdout.fileno() )
        self.old_stderr_fileno = os.dup ( sys.stderr.fileno() )

        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr

        os.dup2 ( self.outnull_file.fileno(), self.old_stdout_fileno_undup )
        os.dup2 ( self.errnull_file.fileno(), self.old_stderr_fileno_undup )

        sys.stdout = self.outnull_file        
        sys.stderr = self.errnull_file
        return self

    def __exit__(self, *_):        
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        os.dup2 ( self.old_stdout_fileno, self.old_stdout_fileno_undup )
        os.dup2 ( self.old_stderr_fileno, self.old_stderr_fileno_undup )

        os.close ( self.old_stdout_fileno )
        os.close ( self.old_stderr_fileno )

        self.outnull_file.close()
        self.errnull_file.close()


def read_ndjson_file(input_path: Path | str) -> list[dict]:
    if not isinstance(input_path, Path):
        input_path = Path(input_path)

    data = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            data.append(json.loads(line))

    return data


def read_json_file(input_path: Path | str) -> dict:
    with open(input_path, "r") as f:
        data = json.load(f)
    return data


def write_ndjson_file(data: list[dict], output_path: Path | str):
    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        for line in data:
            f.write(json.dumps(line) + "\n")


def write_json_file(data: dict, output_path: Path | str):
    if not isinstance(output_path, Path):
        output_path = Path(output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(data) + "\n")


def read_lines(input_file: Path | str) -> list[str]:
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return lines


def read_text_file(input_file: Path | str) -> str:
    with open(input_file, "r", encoding="utf-8") as f:
        text = "".join(list(f.readlines()))
    return text


def write_text_file(text: str, output_file: Path | str) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)


def write_list_to_text_file(lst: list[str], output_file: Path | str, linebreak=True) -> None:
    with open(output_file, "w", encoding="utf-8") as f:
        for line in lst:
            f.write(line)
            if linebreak:
                f.write("\n")


def normalize_name(s):
    return unicodedata.normalize('NFD', s)
