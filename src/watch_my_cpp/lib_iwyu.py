import os
import re

CORRECT_RE = re.compile(r"^\((.*?) has correct #includes/fwd-decls\)$")
SHOULD_ADD_RE = re.compile(r"^(.*?) should add these lines:$")
ADD_RE = re.compile("^(.*?) +// (.*)$")
SHOULD_REMOVE_RE = re.compile(r"^(.*?) should remove these lines:$")
FULL_LIST_RE = re.compile(r"The full include-list for (.*?):$")
END_RE = re.compile(r"^---$")
LINES_RE = re.compile(r"^- (.*?)  // lines ([0-9]+)-[0-9]+$")


GENERAL, ADD, REMOVE, LIST = range(4)


def clang_formatter(output, style):
    """Process iwyu's output into something clang-like."""
    formatted = []

    state = (GENERAL, None)
    for line in output.splitlines():
        match = CORRECT_RE.match(line)
        if match:
            formatted.append(
                "%s:1:1: note: #includes/fwd-decls are correct" % match.groups(1)
            )
            continue
        match = SHOULD_ADD_RE.match(line)
        if match:
            state = (ADD, match.group(1))
            continue
        match = SHOULD_REMOVE_RE.match(line)
        if match:
            state = (REMOVE, match.group(1))
            continue
        match = FULL_LIST_RE.match(line)
        if match:
            state = (LIST, match.group(1))
        elif END_RE.match(line):
            state = (GENERAL, None)
        elif not line.strip():
            continue
        elif state[0] == GENERAL:
            formatted.append(line)
        elif state[0] == ADD:
            match = ADD_RE.match(line)
            if match:
                formatted.append(
                    "%s:1:1: %s: add '%s' (%s)"
                    % (state[1], style, match.group(1), match.group(2))
                )
            else:
                formatted.append("%s:1:1: %s: add '%s'" % (state[1], style, line))
        elif state[0] == REMOVE:
            match = LINES_RE.match(line)
            if match:
                line_no = match.group(2) if match else "1"
                formatted.append(
                    "%s:%s:1: %s: superfluous '%s'"
                    % (state[1], line_no, style, match.group(1))
                )

    return os.linesep.join(formatted)


DEFAULT_FORMAT = "iwyu"

FORMATTERS = {
    "iwyu": lambda output: output,
    "clang": lambda output: clang_formatter(output, style="error"),
    "clang-warning": lambda output: clang_formatter(output, style="warning"),
}
