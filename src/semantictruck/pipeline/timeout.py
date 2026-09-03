"""A wall-clock limit for one run.

`SIGALRM` rather than a subprocess, because the campaign wants to record a slow setting and carry
on with the loaded cloud and the warm Stage 1 cache rather than tear the interpreter down.

**It cannot interrupt a call that is inside Open3D's C++.** The signal is delivered but only runs
its handler when the interpreter next reaches a bytecode boundary, so a reconstruction that has
disappeared into the library will run to completion however long that takes. This is not a
hypothetical: ball pivoting at radii approaching the object scale has taken over half an hour on a
single load and was recorded as a degenerate mesh rather than as a timeout. The sweep's strike rule
is the defence against that, not this.
"""

from __future__ import annotations

import signal

DEFAULT_TIMEOUT_S = 900


class Timeout(Exception):
    """Raised in the main thread when a run exceeds its limit."""


def _raise_timeout(signum, frame):
    raise Timeout()


class time_limit:
    """Context manager enforcing a wall-clock limit. A no-op when `seconds` is falsy."""

    def __init__(self, seconds: int):
        self.seconds = int(seconds or 0)

    def __enter__(self):
        if self.seconds:
            signal.signal(signal.SIGALRM, _raise_timeout)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, *exc):
        if self.seconds:
            signal.alarm(0)
        return False
