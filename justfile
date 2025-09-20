default:
    @just --list

build:
    uv build

run-dev *args:
    uv run watch-my-cpp -v {{ args }}

format:
    treefmt
