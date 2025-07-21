import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePath
from typing import Iterable

DEFAULT_MAX_QUEUE = 10000


class CompilerFamily(StrEnum):
    CLANG = "clang"
    GCC = "GCC"


class DiagnosticSource(StrEnum):
    COMPILER = "compiler"
    CLANG_TIDY = "clang-tidy"
    IWYU = "IWYU"
    CPPCHECK = "cppcheck"


@dataclass
class Diagnostic:
    file: str
    file_short: str  # a compressed version of `file` for display purposes
    line: int
    column: int
    level: str
    message: str
    categories: list[str]
    source: str


def find_clang_tidy():
    """Find clang-tidy executable and return its full pathname."""
    env_clang_tidy_path = os.environ.get("CLANG_TIDY_BINARY")
    if env_clang_tidy_path:
        return os.path.realpath(env_clang_tidy_path)

    # Search in same dir as this script.
    clang_tidy_path = shutil.which("clang-tidy", path=os.path.dirname(__file__))
    if clang_tidy_path:
        return os.path.realpath(clang_tidy_path)

    # Search the system PATH.
    clang_tidy_path = shutil.which("clang-tidy")
    if clang_tidy_path:
        return os.path.realpath(clang_tidy_path)

    return None


def find_include_what_you_use():
    """Find IWYU executable and return its full pathname."""
    env_iwyu_path = os.environ.get("IWYU_BINARY")
    if env_iwyu_path:
        return os.path.realpath(env_iwyu_path)

    # Search in same dir as this script.
    iwyu_path = shutil.which("include-what-you-use", path=os.path.dirname(__file__))
    if iwyu_path:
        return os.path.realpath(iwyu_path)

    # Search the system PATH.
    iwyu_path = shutil.which("include-what-you-use")
    if iwyu_path:
        return os.path.realpath(iwyu_path)

    return None


def find_cppcheck():
    """Find cppcheck executable and return its full pathname."""
    env_cppcheck_path = os.environ.get("CPPCHECK_BINARY")
    if env_cppcheck_path:
        return os.path.realpath(env_cppcheck_path)

    # Search in same dir as this script.
    cppcheck_path = shutil.which("cppcheck", path=os.path.dirname(__file__))
    if cppcheck_path:
        return os.path.realpath(cppcheck_path)

    # Search the system PATH.
    cppcheck_path = shutil.which("cppcheck")
    if cppcheck_path:
        return os.path.realpath(cppcheck_path)

    return None


def detect_compiler_family(cxx_exe: str) -> CompilerFamily:
    r = subprocess.run([cxx_exe, "--version"], capture_output=True)
    output = r.stdout.decode()
    if "clang" in output:
        return CompilerFamily.CLANG
    if "GCC" in output:
        return CompilerFamily.GCC
    raise RuntimeError(f"Failed to detect compiler family from output: {output}")


_is_relevant_header_cache = {}


def is_relevant_header(header_file: str, include_dirs: Iterable[str]) -> bool:
    if header_file in _is_relevant_header_cache:
        return _is_relevant_header_cache[header_file]

    def go(file_name: str):
        if is_header_file(file_name):
            for incl_dir in include_dirs:
                incl_dir = os.path.normpath(incl_dir)
                cp = os.path.commonpath([os.path.normpath(file_name), incl_dir])
                if cp == incl_dir:
                    return True

        return False

    result = go(header_file)

    _is_relevant_header_cache[header_file] = result

    return result


def is_path_under(path: str, directory: str) -> bool:
    try:
        Path(path).resolve().relative_to(Path(directory).resolve())
        return True
    except ValueError:
        return False


def compress_path(path: str) -> str:
    h, t = os.path.split(path)
    p = PurePath(h)
    segs = [part[0] for part in p.parts]
    segs.append(t)

    return os.path.join(*segs)


def extract_include_paths(compile_cmd: str) -> list[str]:
    # Matches:
    # -I/path/to/include
    # -I /path/to/include
    # -isystem /another/path
    pattern = r"(?:-I\s*|-isystem\s+)([^\s]+)"
    return re.findall(pattern, compile_cmd)


def extract_diagnostic(line: str, source: DiagnosticSource) -> Diagnostic | None:
    # Example diagnostic line with category
    # line = "t.c:3:11: warning: conversion specifies type 'char *' but the argument has type 'int' [-Wformat,Format String]"

    # Regular expression to capture file, line, column, level, message, and category
    pattern = re.compile(r"^(.*?):(\d+):(\d+):\s+(\w+):\s+(.*?)(?:\s+\[([^\]]+)\])?$")

    match = pattern.match(line)
    if match:
        file, line_num, col_num, level, message, category = match.groups()
        return Diagnostic(
            file=file,
            file_short=file,
            line=int(line_num),
            column=int(col_num),
            level=level,
            message=message,
            categories=category.split(",") if category else [],
            source=source,
        )

    else:
        return None


def is_header_file(file_name: str) -> bool:
    return file_name.endswith(".h") or file_name.endswith(".hpp")


def split_list(lst: list, max_size: int):
    return [lst[i : i + max_size] for i in range(0, len(lst), max_size)]
