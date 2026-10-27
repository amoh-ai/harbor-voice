"""Shared pytest setup for harbor-voice tests."""

import asyncio
import os
import sys

import pytest

# Tests invoke harbor's CLI and job code; make sure they never emit telemetry.
os.environ["HARBOR_TELEMETRY"] = "off"


@pytest.fixture(scope="session")
def event_loop_policy():
    """ProactorEventLoop on Windows, which asyncio subprocesses need."""
    if sys.platform == "win32":
        return asyncio.WindowsProactorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()
