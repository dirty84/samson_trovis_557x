"""Tests for the resynchronising TROVIS read path (modbus_link.py).

Run without Home Assistant:

    pip install modbus-connection pytest pytest-asyncio
    pytest tests/test_modbus_link.py
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest
from modbus_connection import (
    ModbusDesyncError,
    ModbusProtocolError,
    ModbusTimeoutError,
)

# Load the module by path: the package __init__ needs Home Assistant.
_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "custom_components"
    / "samson_trovis_557x"
    / "modbus_link.py"
)
_SPEC = importlib.util.spec_from_file_location("trovis_modbus_link", _PATH)
assert _SPEC and _SPEC.loader
link = importlib.util.module_from_spec(_SPEC)
sys.modules["trovis_modbus_link"] = link
_SPEC.loader.exec_module(link)


async def _no_sleep(_: float) -> None:
    """Skip real settle delays in tests."""


def _registers(address: int, count: int) -> list[int]:
    """Deterministic register content: value == address."""
    return list(range(address, address + count))


class SlowRtuUnit:
    """Fake RTU unit that reproduces the late-reply desync seen on FW 2.00.

    The first request in ``slow_calls`` times out, but its reply is delivered
    late into the receive buffer. Without a resync, the *next* request
    consumes that stale reply. ``disconnect()`` clears the buffer, like a
    reconnect with fresh serial buffers does.
    """

    def __init__(self, slow_calls: set[tuple[int, int]]) -> None:
        self._slow_calls = set(slow_calls)
        self._rx_buffer: list[list[int]] = []
        self.calls: list[tuple[int, int]] = []
        self.disconnects = 0
        self.required_timeouts: list[float | None] = []
        self.writes: list[tuple[int, int]] = []

    def require_timeout(self, seconds: float | None) -> None:
        self.required_timeouts.append(seconds)

    async def disconnect(self) -> None:
        self.disconnects += 1
        self._rx_buffer.clear()

    async def read_holding_registers(self, address: int, count: int) -> list[int]:
        self.calls.append((address, count))
        if self._rx_buffer:
            stale = self._rx_buffer.pop(0)
            if len(stale) != count:
                raise ModbusProtocolError(
                    f"read_holding_registers({address}, {count}): Invalid "
                    f"register count: expected {count}, got {len(stale)}"
                )
            return stale
        if (address, count) in self._slow_calls:
            self._slow_calls.discard((address, count))
            self._rx_buffer.append(_registers(address, count))
            raise ModbusTimeoutError(
                f"read_holding_registers({address}, {count}): Response timeout"
            )
        return _registers(address, count)

    async def read_coils(self, address: int, count: int) -> list[bool]:
        # Some backends pad coil reads to full bytes.
        return [False] * (((count + 7) // 8) * 8)

    async def write_register(self, address: int, value: int) -> None:
        self.writes.append((address, value))


def _wrap(unit: object) -> link.ReadRetryModbusUnit:
    return link.ReadRetryModbusUnit(unit, "test", sleep=_no_sleep)


@pytest.mark.asyncio
async def test_late_reply_does_not_shift_following_reads() -> None:
    """The FW 2.00 failure: (0,6) times out, (9,22) must still be correct."""
    unit = SlowRtuUnit(slow_calls={(0, 6)})
    wrapped = _wrap(unit)

    assert await wrapped.read_holding_registers(0, 6) == _registers(0, 6)
    assert await wrapped.read_holding_registers(9, 22) == _registers(9, 22)
    assert unit.disconnects == 1
    assert wrapped.resync_count == 1


@pytest.mark.asyncio
async def test_timeouts_exhausted_raise() -> None:
    class AlwaysTimeout(SlowRtuUnit):
        async def read_holding_registers(self, address: int, count: int):
            self.calls.append((address, count))
            raise ModbusTimeoutError("timeout")

    unit = AlwaysTimeout(set())
    with pytest.raises(ModbusTimeoutError):
        await _wrap(unit).read_holding_registers(0, 1)
    assert len(unit.calls) == link.READ_RETRIES + 1
    assert unit.disconnects == link.READ_RETRIES


@pytest.mark.asyncio
async def test_repeated_unusable_reply_reaches_caller() -> None:
    """A reply unusable twice after resync must bubble up (address learning)."""

    class Garbled(SlowRtuUnit):
        async def read_coils(self, address: int, count: int):
            self.calls.append((address, count))
            raise ModbusProtocolError(f"read_coils({address}, {count}): junk")

    unit = Garbled(set())
    with pytest.raises(ModbusProtocolError, match=r"^read_coils\(401, 13\)"):
        await _wrap(unit).read_coils(401, 13)
    assert len(unit.calls) == link.PROTOCOL_RETRIES + 1


@pytest.mark.asyncio
async def test_desync_is_retried_once() -> None:
    class DesyncOnce(SlowRtuUnit):
        def __init__(self) -> None:
            super().__init__(set())
            self._failed = False

        async def read_holding_registers(self, address: int, count: int):
            if not self._failed:
                self._failed = True
                raise ModbusDesyncError("header mismatch")
            return _registers(address, count)

    assert await _wrap(DesyncOnce()).read_holding_registers(5, 3) == [5, 6, 7]


@pytest.mark.asyncio
async def test_short_register_result_is_rejected_then_retried() -> None:
    class ShortOnce(SlowRtuUnit):
        def __init__(self) -> None:
            super().__init__(set())
            self._short = True

        async def read_holding_registers(self, address: int, count: int):
            if self._short:
                self._short = False
                return _registers(address, count - 1)
            return _registers(address, count)

    assert await _wrap(ShortOnce()).read_holding_registers(9, 4) == [9, 10, 11, 12]


@pytest.mark.asyncio
async def test_padded_coils_are_accepted() -> None:
    result = await _wrap(SlowRtuUnit(set())).read_coils(0, 13)
    assert len(result) == 16


def test_timeout_requirement_is_raised_to_minimum() -> None:
    unit = SlowRtuUnit(set())
    wrapped = _wrap(unit)
    wrapped.require_timeout(1)
    wrapped.require_timeout(None)
    wrapped.require_timeout(10)
    assert unit.required_timeouts == [link.MIN_LINK_TIMEOUT, link.MIN_LINK_TIMEOUT, 10.0]


@pytest.mark.asyncio
async def test_writes_are_forwarded_without_retry() -> None:
    unit = SlowRtuUnit(set())
    await _wrap(unit).write_register(100, 5)
    assert unit.writes == [(100, 5)]


def test_wrapping_is_idempotent() -> None:
    wrapped = _wrap(SlowRtuUnit(set()))
    assert link.with_read_retries(wrapped, "again") is wrapped


@pytest.mark.asyncio
async def test_failed_open_is_retried_with_backoff() -> None:
    """'could not open serial port' after a resync must not fail the poll."""
    from modbus_connection import ModbusConnectionError

    class FlakyOpen(SlowRtuUnit):
        def __init__(self) -> None:
            super().__init__(set())
            self._open_failures = 2

        async def read_holding_registers(self, address: int, count: int):
            self.calls.append((address, count))
            if self._open_failures:
                self._open_failures -= 1
                raise ModbusConnectionError("could not open serial port esphome-hass://x")
            return _registers(address, count)

    sleeps: list[float] = []

    async def record(seconds: float) -> None:
        sleeps.append(seconds)

    unit = FlakyOpen()
    wrapped = link.ReadRetryModbusUnit(unit, "test", sleep=record)
    assert await wrapped.read_holding_registers(3, 2) == [3, 4]
    assert sleeps == list(link.CONNECT_BACKOFF_SECONDS[:2])
    assert wrapped.reconnect_count == 2


@pytest.mark.asyncio
async def test_open_failures_exhausted_raise() -> None:
    from modbus_connection import ModbusConnectionError

    class NeverOpens(SlowRtuUnit):
        async def read_holding_registers(self, address: int, count: int):
            self.calls.append((address, count))
            raise ModbusConnectionError("could not open serial port")

    unit = NeverOpens(set())
    with pytest.raises(ModbusConnectionError):
        await _wrap(unit).read_holding_registers(0, 1)
    assert len(unit.calls) == len(link.CONNECT_BACKOFF_SECONDS) + 1


def test_link_grace_bridges_only_within_window() -> None:
    grace = link.LinkGrace(grace_seconds=180)
    assert grace.bridge(10.0) is False  # no good poll yet: never bridge
    grace.success(100.0)
    assert grace.bridge(160.0) is True
    assert grace.bridge(280.0) is True  # exactly at the limit
    assert grace.bridge(281.0) is False  # older than the window: surface it
    assert grace.failures == 2
    assert grace.success(300.0) == 2
    assert grace.failures == 0
