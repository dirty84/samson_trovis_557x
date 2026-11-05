"""Tests for the setup guard (setup_guard.py).

Run without Home Assistant:

    pytest tests/test_setup_guard.py
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
from types import SimpleNamespace

# Load the module by path: the package __init__ needs Home Assistant.
_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "custom_components"
    / "samson_trovis_557x"
    / "setup_guard.py"
)
_SPEC = importlib.util.spec_from_file_location("trovis_setup_guard", _PATH)
assert _SPEC and _SPEC.loader
guard = importlib.util.module_from_spec(_SPEC)
sys.modules["trovis_setup_guard"] = guard
_SPEC.loader.exec_module(guard)


def _device(system_code: float | None) -> SimpleNamespace:
    return SimpleNamespace(info=SimpleNamespace(system_code=system_code))


def test_known_system_code_proceeds() -> None:
    assert guard.setup_not_ready_reason(_device(1.0), last_update_success=True) is None


def test_known_system_code_proceeds_even_after_failed_poll() -> None:
    # Identity read, a later optional block failed: entry must still load.
    assert guard.setup_not_ready_reason(_device(1.0), last_update_success=False) is None


def test_link_down_at_setup_requests_retry() -> None:
    reason = guard.setup_not_ready_reason(_device(None), last_update_success=False)
    assert reason is not None
    assert "system code unknown" in reason


def test_successful_poll_without_identity_does_not_loop() -> None:
    # Identity block excluded/empty but link fine: retrying would not help.
    assert guard.setup_not_ready_reason(_device(None), last_update_success=True) is None


def test_missing_info_object_counts_as_unknown() -> None:
    assert not guard.controller_identity_known(SimpleNamespace())
    assert not guard.controller_identity_known(SimpleNamespace(info=None))


def test_zero_system_code_counts_as_known() -> None:
    # 0.0 is falsy but a valid reading; must not be treated as missing.
    assert guard.controller_identity_known(_device(0.0))
