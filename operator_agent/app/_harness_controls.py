"""Process-local controls reserved for tests and benchmark harnesses.

These switches intentionally have no environment-variable or CLI input.  A test or
benchmark may enable them in-process before constructing the agent in order to
redirect/disable side-effect stores or use an oversized heartbeat interval.  A
normal task, subprocess, or candidate cannot turn these controls on by setting an
environment variable.
"""

_ENABLED = False


def enable_for_harness() -> None:
    """Enable harness-only test controls for this Python process."""
    global _ENABLED
    _ENABLED = True


def enabled() -> bool:
    """Return whether harness-only controls are enabled in this process."""
    return _ENABLED
