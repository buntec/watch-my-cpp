import argparse
import os
import subprocess

from lib import DEFAULT_MAX_QUEUE


def main():
    parser = argparse.ArgumentParser(
        description="Given a `compile_commands.json` file, watch-my-cpp compiles your C/C++ sources in the background to provide live diagnostics in a web based UI. Changes to sources and header files are detected and trigger a recompilation of the relevant sources."
    )

    parser.add_argument(
        "--workers",
        "-j",
        type=int,
        help="max degree of compile concurrency",
    )
    parser.add_argument(
        "--max_queue",
        "-q",
        type=int,
        default=DEFAULT_MAX_QUEUE,
        help=f"max number of compile jobs in queue, defaults to {DEFAULT_MAX_QUEUE}",
    )
    parser.add_argument(
        "compile_commands_path",
        metavar="PATH",
        help="path to compile_commands.json - either a path to the file or a directory containing it",
    )

    parser.add_argument(
        "--ignore-patterns",
        "-i",
        nargs="*",
        help="zero or more shell-style wildcard patterns of file paths to ignore",
    )

    parser.add_argument(
        "--include-patterns",
        "-g",
        nargs="*",
        help="zero or more shell-style wildcard patterns of file paths to include",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="increase verbosity (can be used multiple times)",
    )

    parser.add_argument("--dev", action="store_true", help="run fastapi in dev mode")

    args = parser.parse_args()

    env = os.environ

    env.update(
        {
            "WMCPP_COMPILE_COMMANDS_PATH": os.path.abspath(args.compile_commands_path),
            "WMCPP_MAX_QUEUE": str(args.max_queue),
            "WMCPP_VERBOSITY": str(args.verbose),
            # "CCACHE_DEBUGLEVEL": "1",
            # "CCACHE_DEBUG": "1",
        }
    )

    if args.workers:
        env.update(
            {
                "WMCPP_WORKERS": str(args.workers),
            }
        )

    if args.ignore_patterns:
        env.update(
            {
                "WMCPP_IGNORE_PATTERNS": ";".join(args.ignore_patterns),
            }
        )

    if args.include_patterns:
        env.update(
            {
                "WMCPP_INCLUDE_PATTERNS": ";".join(args.include_patterns),
            }
        )

    cmd = ["uvicorn"]

    cmd.extend(["--host", "0.0.0.0", "--log-level", "info"])

    if args.dev:
        cmd.append("--reload")

    cmd.append("main:app")

    subprocess.run(cmd, env=env)


if __name__ == "__main__":
    main()
