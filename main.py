import asyncio
import copy
import fnmatch
import itertools
import json
import logging
import os
import sys
import tempfile
import time
import uuid
from asyncio import Event, Queue, QueueEmpty
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, AsyncGenerator, Iterable, Optional

import watchfiles
from pydantic_settings import BaseSettings, SettingsConfigDict
from rich.logging import RichHandler
from starlette.applications import Starlette
from starlette.routing import Mount, WebSocketRoute
from starlette.staticfiles import StaticFiles
from starlette.websockets import WebSocket
from watchfiles import awatch

import lib_iwyu
from lib import (
    DEFAULT_MAX_QUEUE,
    CompilerFamily,
    Diagnostic,
    DiagnosticSource,
    compress_path,
    detect_compiler_family,
    extract_diagnostic,
    extract_include_paths,
    find_clang_tidy,
    find_include_what_you_use,
    is_relevant_header,
)

COMPILER_OPTIONS: dict[CompilerFamily, list[str]] = {
    CompilerFamily.CLANG: [
        "-O0",
        "-fno-caret-diagnostics",
        "-fno-color-diagnostics",
    ],
    CompilerFamily.GCC: [
        "-O0",
        "-fno-diagnostics-show-caret",
        "-fno-diagnostics-show-line-numbers",
        "-fdiagnostics-color=never",
        "-fdiagnostics-urls=never",
        "-fdiagnostics-path-format=separate-events",
        "-fdiagnostics-text-art-charset=none",
    ],
}

logger = logging.getLogger(__name__)

# compilation is cpu bound so having more workers than cores doesn't really make sense
MAX_WORKERS: int = os.cpu_count() or 1

CLANG_TIDY_EXECUTABLE = find_clang_tidy()

IWYU_EXECUTABLE = find_include_what_you_use()


class Settings(BaseSettings):
    compile_commands_path: str = ""
    max_queue: int = DEFAULT_MAX_QUEUE
    workers: int = MAX_WORKERS // 2
    max_workers: int = MAX_WORKERS
    ignore_patterns: Optional[str] = None  # semicolon separated list of glob patterns
    include_patterns: Optional[str] = None  # semicolon separated list of glob patterns
    verbosity: int = 0
    clang_tidy: bool = False
    iwyu: bool = False

    model_config = SettingsConfigDict(env_prefix="wmcpp_")


class SourceFileStatus(StrEnum):
    COMPILING = "compiling"
    COMPILED = "compiled"
    NEW = "new"


@dataclass
class SourceFile:
    path: str
    status: str = SourceFileStatus.NEW
    compile_time: float | None = None
    compile_timestamp: str | None = None  # string for ease of serialization
    diagnostics: list[Diagnostic] | None = None


@dataclass
class MsgCompile:
    file: str
    force_clang_tidy: bool = False
    force_iwyu: bool = False


class NotificationKind(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Notification:
    message: str
    kind: NotificationKind


@dataclass
class MsgSourceFile:
    file: SourceFile
    timestamp: float
    type: str = "file"


@dataclass
class MsgFileDeleted:
    file: str
    type: str = "file_deleted"


@dataclass
class MsgNotifications:
    notifications: list[Notification]
    type: str = "notifications"


@dataclass
class Status:
    settings: dict
    paused: bool
    n_sources: int
    n_queued: int
    n_queued_low_prio: int
    temp_dir: str


@dataclass
class MsgStatus:
    status: Status
    type: str = "status"


type MsgServer = MsgFileDeleted | MsgSourceFile | MsgNotifications | MsgStatus


# source files are relative to `common_prefix`
# include paths are always absolute
@dataclass
class State:
    common_prefix: str = ""
    include_paths: set[str] = field(default_factory=set)
    include_paths_by_source: dict[str, set[str]] = field(default_factory=dict)
    commands: dict[str, str] = field(default_factory=dict)
    iwyu_commands: dict[str, str] = field(default_factory=dict)
    source_files: dict[str, SourceFile] = field(default_factory=dict)
    temp_dir: str = ""


state = State()

settings = Settings()


# queues
q_compile: Queue[MsgCompile] = Queue(settings.max_queue)
q_compile_low_prio: Queue[MsgCompile] = Queue(settings.max_queue)
q_ws_client_actions: Queue[dict[str, Any]] = Queue(100)
qs_tx: dict[str, Queue[MsgServer]] = {}  # WS send queues
qs_rx: dict = {}  # WS receive queues

# events
ev_stats_update = Event()
ev_settings_change = Event()
ev_resume = Event()
ev_kill_switch = Event()
ev_num_workers_change = Event()

set_compile: set[str] = set()
set_compile_low_prio: set[str] = set()


class KillSwitchFlipped(Exception):
    """Exception raised to trigger a logical restart of the service"""


class ReloadCompileCommands(Exception):
    """Exception raised to trigger a reload of compile_commands.json."""


async def ws_broadcast(msg: MsgServer):
    for _, q in qs_tx.items():
        await q.put(msg)


async def broadcast_notification(msg: str, kind: NotificationKind):
    await ws_broadcast(MsgNotifications([Notification(msg, kind)]))


async def send_file(file: SourceFile):
    await ws_broadcast(MsgSourceFile(file, time.time()))


async def raise_on_kill_switch():
    await ev_kill_switch.wait()
    raise KillSwitchFlipped()


async def handle_client_actions():
    while True:
        msg = await q_ws_client_actions.get()
        match msg:
            case {"type": "heartbeat"}:
                logger.debug("heartbeat received")
            case {"type": "pause"}:
                ev_resume.clear()
            case {"type": "resume"}:
                ev_resume.set()
            case {"type": "toggle_clang_tidy"}:
                settings.clang_tidy = not settings.clang_tidy
                ev_settings_change.set()
            case {"type": "toggle_iwyu"}:
                settings.iwyu = not settings.iwyu
                ev_settings_change.set()
            case {"type": "recompile_file", "file": file}:
                logger.info(f"recompile file action received for {file}")
                if file not in set_compile:
                    set_compile.add(file)
                    await q_compile.put(MsgCompile(file))
            case {"type": "recompile_file_and_force_clang_tidy", "file": file}:
                logger.info(f"recompile file action (+ clang-tidy) received for {file}")
                set_compile.add(file)
                await q_compile.put(MsgCompile(file, force_clang_tidy=True))
            case {"type": "recompile_file_and_force_iwyu", "file": file}:
                logger.info(f"recompile file action (+ iwyu) received for {file}")
                set_compile.add(file)
                await q_compile.put(MsgCompile(file, force_iwyu=True))
            case {"type": "kill_switch"}:
                ev_kill_switch.set()
            case {"type": "num_workers_change", "n": n}:
                if n < 0 or n > MAX_WORKERS:
                    raise ValueError(
                        f"#workers should be between 0 and {MAX_WORKERS} but was {n}"
                    )
                logger.info(f"Setting # workers to {n}")
                settings.workers = n
                ev_num_workers_change.set()
                ev_num_workers_change.clear()
            case _:
                logger.warning(f"unexpected message: {msg}")


def wrap_compile_command(cmd: str, cxx_family: CompilerFamily) -> str:
    cxx_options = COMPILER_OPTIONS[cxx_family]
    return f"ccache {cmd} {' '.join(cxx_options)}"


def make_iwyu_command(compile_cmd: str) -> str:
    args = [IWYU_EXECUTABLE or "include-what-you-use"]
    args.extend(compile_cmd.split(" ")[1:])
    return " ".join(args)


_get_impacted_source_files_cache = {}


def get_impacted_source_files(header_file: str) -> list[str]:
    if not is_relevant_header(header_file, state.include_paths):
        return []

    if header_file in _get_impacted_source_files_cache:
        return _get_impacted_source_files_cache[header_file]

    def go(header_file):
        result = []
        for source_file in state.source_files.keys():
            incl_paths = state.include_paths_by_source[source_file]
            for incl_path in incl_paths:
                # check whether header_file is under incl_path
                common_prefix = os.path.commonpath([header_file, incl_path])
                if common_prefix == incl_path:
                    result.append(source_file)
                    break  # add source file at most once
        return result

    result = go(header_file)

    _get_impacted_source_files_cache[header_file] = result

    return result


def diagnostics_summary(source_files: dict[str, SourceFile]):
    errors: int = 0
    warnings: int = 0
    notes: int = 0

    for sf in source_files.values():
        if sf.diagnostics:
            for diag in sf.diagnostics:
                match diag.level:
                    case "error":
                        errors += 1
                    case "warning":
                        warnings += 1
                    case "note":
                        notes += 1
    return {
        "errors": errors,
        "warnings": warnings,
        "notes": notes,
    }


async def run_metrics_reporter():
    while True:
        await ev_stats_update.wait()
        logger.info(f"""---- metrics ----
items in compile queue: {q_compile.qsize()}
items in low-prio compile queue: {q_compile_low_prio.qsize()}
source files: {len(state.source_files)}
diagnostics: {diagnostics_summary(state.source_files)}""")
        ev_stats_update.clear()
        await asyncio.sleep(3.0)  # debounce


async def compile_files_low_prio(files: Iterable[str]):
    for file_path in files:
        if file_path not in set_compile_low_prio:
            set_compile_low_prio.add(file_path)
            await q_compile_low_prio.put(MsgCompile(file_path))


def settings_change_should_recompile(old: Settings, new: Settings) -> bool:
    """Return True if the given change of settings requires a recompilation of all sources."""
    flags_old = [old.iwyu, old.clang_tidy]
    flags_new = [new.iwyu, new.clang_tidy]
    return any((not o) and n for o, n in zip(flags_old, flags_new))


async def handle_settings_change():
    while True:
        settings_old = copy.deepcopy(settings)
        await ev_settings_change.wait()
        if settings_change_should_recompile(settings_old, settings):
            logger.info("Settings change requires recompilation of all sources...")
            await compile_files_low_prio(state.source_files.keys())

        ev_settings_change.clear()
        await asyncio.sleep(1.0)  # debounce


async def watch_compile_commands(path_to_compile_commands: str):
    async for changes in awatch(path_to_compile_commands, recursive=False):
        for change in changes:
            change_type, file_path = change
            if change_type == watchfiles.Change.deleted:
                # if compile_commands.json is deleted, we don't do anything and wait for it to be re-created,
                # thus triggering another change type
                continue
            if file_path.endswith("compile_commands.json"):
                logger.warning(f"change in {file_path} detected")
                await broadcast_notification(
                    "Change detected in <code>compile_commands.json</code> - reloading...",
                    NotificationKind.INFO,
                )
                raise ReloadCompileCommands()


async def watch_files():
    async for changes in awatch(state.common_prefix, recursive=True):
        for change in changes:
            change_type, file_path_abs = change
            file_path = os.path.relpath(file_path_abs, state.common_prefix)
            if change_type == watchfiles.Change.deleted:
                if file_path in state.source_files:
                    await ws_broadcast(MsgFileDeleted(file_path))
                    await broadcast_notification(
                        f"Deletion of source <code>{compress_path(file_path)}</code> detected.",
                        NotificationKind.WARNING,
                    )
                continue
            if file_path in state.source_files:
                logger.debug(f"change detected in source file {file_path}")
                await broadcast_notification(
                    f"Change detected in <code>{compress_path(file_path)}</code>",
                    NotificationKind.INFO,
                )
                if change_type == watchfiles.Change.added:
                    await ws_broadcast(
                        MsgSourceFile(SourceFile(file_path), time.time())
                    )

                if file_path not in set_compile:
                    set_compile.add(file_path)
                    await q_compile.put(MsgCompile(file_path))
            elif is_relevant_header(file_path_abs, state.include_paths):
                logger.debug(f"change detected in header file: {file_path_abs}")
                await compile_files_low_prio(get_impacted_source_files(file_path_abs))


async def stream_compile_messages() -> AsyncGenerator[MsgCompile]:
    while True:
        await ev_resume.wait()
        try:
            msg = q_compile.get_nowait()
            set_compile.remove(msg.file)
            q_compile.task_done()
            yield msg
        except QueueEmpty:  # if high prio queue is empty, try low prio queue
            try:
                msg = await asyncio.wait_for(q_compile_low_prio.get(), 0.1)
                set_compile_low_prio.remove(msg.file)
                q_compile_low_prio.task_done()
                yield msg
            except TimeoutError:
                continue


async def wait_if_index_exceeds_num_workers(index: int):
    if index >= settings.workers:
        while index >= settings.workers:
            logger.info(f"worker-{index}: going to sleep.")
            await ev_num_workers_change.wait()
            await asyncio.sleep(0.1)
        logger.info(f"worker-{index}: resuming work.")


async def worker(index: int):
    await wait_if_index_exceeds_num_workers(index)

    async for msg in stream_compile_messages():
        file = msg.file

        if file not in state.commands:
            logger.warning(
                f"worker-{index} failed to find compile command for {file} - skipping..."
            )
            continue

        cmd = state.commands[file]

        sf = state.source_files[file]

        sf.diagnostics = None
        sf.status = SourceFileStatus.COMPILING

        await send_file(sf)

        logger.debug(f"worker-{index}: compiling {file}...")

        logger.debug(f"compiler command: {cmd}")

        t0 = time.perf_counter()

        # time should suffice, date seems overkill "%Y-%m-%d %H:%M:%S"
        sf.compile_timestamp = datetime.now().strftime("%H:%M:%S")

        proc = await asyncio.create_subprocess_shell(
            cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        _, stderr_b = await proc.communicate()

        t1 = time.perf_counter()

        dt = t1 - t0

        logger.debug(f"worker-{index}: done compiling {file} in {dt} seconds")

        stderr: str = stderr_b.decode()

        logger.debug(f"compiler output: {stderr}")

        sf.compile_time = dt
        sf.status = SourceFileStatus.COMPILED

        diags: list[Diagnostic] = []

        for line in stderr.splitlines():
            diag = extract_diagnostic(line, DiagnosticSource.COMPILER)
            if diag:
                diag.file = os.path.relpath(diag.file, state.common_prefix)
                if diag.file.startswith("../"):  # system or 3rd party header
                    diag.file_short = os.path.basename(diag.file)
                else:
                    diag.file_short = compress_path(diag.file)
                diags.append(diag)

        if IWYU_EXECUTABLE and (settings.iwyu or msg.force_iwyu):
            cmd = state.iwyu_commands[file]

            logger.debug(f"running iwyu command: {cmd}")

            t0 = time.perf_counter()

            proc = await asyncio.create_subprocess_shell(
                cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            _, stderr_b = await proc.communicate()

            t1 = time.perf_counter()

            dt = t1 - t0

            logger.debug(f"worker-{index}: done iwyu {file} in {dt} seconds")

            stderr: str = stderr_b.decode()

            logger.debug(f"iwyu output: {stderr}")

            stderr = lib_iwyu.FORMATTERS["clang-warning"](stderr)

            # logger.debug(f"iwyu output clang-ified: {stderr}")

            for line in stderr.splitlines():
                diag = extract_diagnostic(line, DiagnosticSource.IWYU)
                if diag:
                    diag.file = os.path.relpath(diag.file, state.common_prefix)
                    if diag.file.startswith("../"):  # system or 3rd party header
                        diag.file_short = os.path.basename(diag.file)
                    else:
                        diag.file_short = compress_path(diag.file)
                    diags.append(diag)

        if CLANG_TIDY_EXECUTABLE and (settings.clang_tidy or msg.force_clang_tidy):
            t0 = time.perf_counter()

            cmd = f"clang-tidy --use-color=false {os.path.join(state.common_prefix, file)}"

            logger.debug(f"running {cmd}")

            proc = await asyncio.create_subprocess_shell(
                cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout_b, stderr_b = await proc.communicate()

            stdout: str = stdout_b.decode()
            stderr: str = stderr_b.decode()

            logger.debug(f"clang-tidy stderr: {stderr}")
            logger.debug(f"clang-tidy stdout: {stdout}")

            t1 = time.perf_counter()

            dt = t1 - t0

            logger.debug(
                f"worker-{index}: running clang-tidy on {file} took {dt} seconds"
            )

            for line in stdout.splitlines():
                diag = extract_diagnostic(line, DiagnosticSource.CLANG_TIDY)
                if diag and diag.level != "error":  # safe to ignore clang-tidy errors
                    diag.file = os.path.relpath(diag.file, state.common_prefix)
                    if diag.file.startswith("../"):  # system or 3rd party header
                        diag.file_short = os.path.basename(diag.file)
                    else:
                        diag.file_short = compress_path(diag.file)
                    diags.append(diag)

        if diags:
            sf.diagnostics = diags

        await send_file(sf)

        ev_stats_update.set()

        await wait_if_index_exceeds_num_workers(index)


async def compile_service():
    # start workers
    tasks: list = [asyncio.create_task(worker(i)) for i in range(MAX_WORKERS)]

    # start file watcher
    t1 = asyncio.create_task(watch_files())

    # compile all sources once
    t2 = asyncio.create_task(compile_files_low_prio(state.source_files.keys()))

    # start the statistics reporter
    t3 = asyncio.create_task(run_metrics_reporter())

    tasks.extend([t1, t2, t3])

    await asyncio.gather(*tasks)


def try_read_compile_commands(path: str):
    if os.path.isfile(path) and os.path.basename(path) == "compile_commands.json":
        with open(path) as f:
            return json.load(f)

    if os.path.isdir(path):
        cc_path = os.path.join(path, "compile_commands.json")
        if os.path.isfile(cc_path):
            with open(cc_path) as f:
                return json.load(f)

    raise RuntimeError(
        f"Unable to find `compile_commands.json` under {path}. Note that other names for this file are not suppored."
    )


def init_state(temp_dir: str):
    cc = try_read_compile_commands(settings.compile_commands_path)

    # include only source files matching one or more include patterns
    if settings.include_patterns:
        patterns = settings.include_patterns.split(";")
        cc = [
            c
            for c in cc
            if any(fnmatch.fnmatch(c["file"], pattern) for pattern in patterns)
        ]

    # remove source files matching one or more ignore patterns
    if settings.ignore_patterns:
        patterns = settings.ignore_patterns.split(";")
        cc = [
            c
            for c in cc
            if not any(fnmatch.fnmatch(c["file"], pattern) for pattern in patterns)
        ]

    include_paths = set()
    include_paths_by_source = {}

    common_prefix = os.path.commonpath([c["file"] for c in cc])

    logger.info(f"using common path prefix: {common_prefix}")

    def get_cxx_family():
        for c in cc:
            file = c["file"]
            if file.endswith(".cpp"):
                cmd = c["command"]
                cxx_exe = cmd.split()[0]
                return detect_compiler_family(cxx_exe)
        raise RuntimeError("failed to detect compiler family")

    cxx_family = get_cxx_family()

    logger.info(f"detected compiler family: {cxx_family}")

    if CLANG_TIDY_EXECUTABLE:
        logger.info(f"using clang-tidy executable: {CLANG_TIDY_EXECUTABLE}")
    else:
        logger.warning("clang-tidy executable not found")

    if IWYU_EXECUTABLE:
        logger.info(f"using include-what-you-use executable: {IWYU_EXECUTABLE}")
    else:
        logger.warning("include-what-you-use executable not found")

    # for every source file do the following:
    # - remove common path prefix
    # - replace original output path with temp dir
    # - extract all include paths passed to the compiler (e.g., -I/foo/bar)
    # - append output formatting options to the compile command
    # - wrap the compile command with ccache

    for c in cc:
        old_file_path = c["file"]
        new_file_path = os.path.relpath(old_file_path, common_prefix)
        c["file"] = new_file_path
        old_out = c["output"]
        new_out = os.path.join(temp_dir, os.path.basename(new_file_path) + ".o")
        cmd_old = c["command"]
        icps = extract_include_paths(cmd_old)
        include_paths_by_source[new_file_path] = icps
        include_paths.update(icps)
        cmd_new = cmd_old.replace(old_out, new_out)
        c["command"] = wrap_compile_command(cmd_new, cxx_family)
        c["iwyu-command"] = make_iwyu_command(cmd_new)

    files = {c["file"]: SourceFile(c["file"]) for c in cc}

    commands = {c["file"]: c["command"] for c in cc}
    iwyu_commands = {c["file"]: c["iwyu-command"] for c in cc}

    global state

    state = State(
        common_prefix=common_prefix,
        include_paths=include_paths,
        include_paths_by_source=include_paths_by_source,
        commands=commands,
        iwyu_commands=iwyu_commands,
        source_files=files,
        temp_dir=temp_dir,
    )


def setup_logger(logger: logging.Logger):
    logger.addHandler(
        RichHandler(
            markup=True,
            log_time_format="[%X]",
            omit_repeated_times=False,
            show_path=False,
        )
    )

    if settings.verbosity < 1:
        logger.setLevel(logging.WARNING)
    elif settings.verbosity < 2:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.DEBUG)


async def ws_consume_messages():
    while True:
        empty = True
        for uid, q in qs_rx.items():
            try:
                msg = q.get_nowait()
                logger.debug(f"message received {uid}: {msg}")
                empty = False
                await q_ws_client_actions.put(json.loads(msg))
            except asyncio.QueueEmpty:
                pass
            except json.JSONDecodeError:
                logger.warning("failed to decode client message to JSON")
        if empty:  # all queues are empty, back off before retrying
            await asyncio.sleep(0.1)


background_tasks = []


@asynccontextmanager
async def lifespan(_):
    setup_logger(logger)

    # create temporary dir for compiler output
    tempdir = tempfile.TemporaryDirectory()

    logger.info(f"Using temp directory: {tempdir.name}")

    cc_path = settings.compile_commands_path

    async def run_services():
        while True:
            try:
                logger.info("Starting services...")
                ev_resume.set()
                ev_kill_switch.clear()
                init_state(tempdir.name)
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(raise_on_kill_switch())
                    tg.create_task(handle_settings_change())
                    tg.create_task(ws_consume_messages())
                    tg.create_task(handle_client_actions())
                    tg.create_task(watch_compile_commands(cc_path))
                    tg.create_task(compile_service())
            except* ReloadCompileCommands:
                logger.info("Restarting to pick up changes in compile_commands.json")
            except* KillSwitchFlipped:
                logger.info("Kill switch flipped: restarting services")
            except* Exception:
                logger.exception("Services failed")
                sys.exit()

    # start services
    services = asyncio.create_task(run_services())

    # store reference to task to keep it running
    background_tasks.append(services)

    yield

    services.cancel()
    tempdir.cleanup()


async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    uid = str(uuid.uuid4())

    logger.info(f"Opening new WS connection: {uid}")

    q_tx: Queue[MsgServer] = asyncio.Queue(10000)
    q_rx = asyncio.Queue(10000)

    qs_tx[uid] = q_tx
    qs_rx[uid] = q_rx

    async def send_status():
        while True:
            try:
                await asyncio.wait_for(ev_settings_change.wait(), 1.0)
            except TimeoutError:
                pass
            await q_tx.put(
                MsgStatus(
                    Status(
                        settings.model_dump(mode="json"),
                        not ev_resume.is_set(),
                        len(state.source_files),
                        q_compile.qsize(),
                        q_compile_low_prio.qsize(),
                        state.temp_dir,
                    )
                )
            )

    async def send_heartbeat():
        for i in itertools.count():
            await asyncio.sleep(5)
            text = json.dumps({"ping": i})
            try:
                await websocket.send_text(text)
            except Exception as e:
                logger.warning(f"WS connection {uid}: sending hearbeat failed: {e}")

    async def send_init_data():
        timestamp = time.time()
        # send all source files to hydrate client
        for file in state.source_files.values():
            await q_tx.put(MsgSourceFile(file, timestamp))

    # send in chunks with a maximum delay (in seconds)
    async def send_loop(max_chunk_size: int, max_delay: float):
        buffer = []
        timeout = False
        while True:
            try:
                msg = await asyncio.wait_for(q_tx.get(), max_delay)
                buffer.append(asdict(msg))
                q_tx.task_done()
            except TimeoutError:
                timeout = True
            if len(buffer) >= max_chunk_size or (timeout and buffer):
                text = json.dumps(buffer)
                await websocket.send_text(text)
                buffer.clear()
                timeout = False

    async def recv_loop():
        while True:
            text = await websocket.receive_text()
            await q_rx.put(text)

    try:
        async with asyncio.TaskGroup() as tg:
            tg.create_task(raise_on_kill_switch())
            tg.create_task(send_status())
            tg.create_task(send_heartbeat())
            tg.create_task(send_init_data())
            tg.create_task(send_loop(max_chunk_size=100, max_delay=0.1))
            tg.create_task(recv_loop())
    except* Exception as e:
        logger.info(f"WS connection {uid} exception in task group: {e.exceptions}")
        await websocket.close(1011)
    finally:
        del qs_tx[uid]
        del qs_rx[uid]
        logger.info(f"Closing WS connection {uid}")


routes = [
    WebSocketRoute("/ws", websocket_endpoint),
    Mount("/", app=StaticFiles(directory="static", html=True), name="static"),
]


app = Starlette(routes=routes, lifespan=lifespan)
