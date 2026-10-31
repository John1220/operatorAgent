"""One monotonic epoch across processes, including macOS Python 3.9.

Python <3.10 on macOS offsets time.monotonic() separately per process. POSIX
CLOCK_MONOTONIC is system-wide; Windows' Python monotonic clock is shared too.
"""
import time


def monotonic():
    if hasattr(time, 'clock_gettime') and hasattr(time, 'CLOCK_MONOTONIC'):
        return time.clock_gettime(time.CLOCK_MONOTONIC)
    return time.monotonic()
