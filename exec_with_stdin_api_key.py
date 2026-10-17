#!/usr/bin/env python3
"""Exec a command with a private API key delivered through an inherited pipe FD."""
import os
import sys

MAX_KEY_BYTES = 4096


def _write_all(fd, data):
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv:
        print('usage: exec_with_stdin_api_key.py COMMAND [ARG ...]', file=sys.stderr)
        return 2
    raw = sys.stdin.buffer.read(MAX_KEY_BYTES + 1)
    if len(raw) > MAX_KEY_BYTES or any(char in raw for char in (b'\0', b'\r', b'\n')):
        print('invalid API key input', file=sys.stderr)
        return 2
    try:
        raw.decode('utf-8')
    except UnicodeDecodeError:
        print('API key input must be UTF-8', file=sys.stderr)
        return 2

    environment = os.environ.copy()
    environment.pop('STRATA_API_KEY', None)
    environment.pop('STRATA_API_KEY_FD', None)
    read_fd = write_fd = None
    try:
        if raw:
            read_fd, write_fd = os.pipe()
            _write_all(write_fd, raw)
            os.close(write_fd)
            write_fd = None
            os.set_inheritable(read_fd, True)
            environment['STRATA_API_KEY_FD'] = str(read_fd)
        null_fd = os.open('/dev/null', os.O_RDONLY)
        os.dup2(null_fd, 0)
        os.close(null_fd)
        os.execvpe(argv[0], argv, environment)
    except OSError as exc:
        print(f'cannot exec requested command: {exc}', file=sys.stderr)
        return 127
    finally:
        if read_fd is not None:
            try:
                os.close(read_fd)
            except OSError:
                pass
        if write_fd is not None:
            try:
                os.close(write_fd)
            except OSError:
                pass
    return 127


if __name__ == '__main__':
    raise SystemExit(main())
