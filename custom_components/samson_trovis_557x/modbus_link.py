"""Hardened read path for slow TROVIS links.

Why this exists
---------------
A TROVIS 5573 with firmware 2.00 behind an ESPHome serial proxy answers
noticeably slower than the 1 s per-request timeout that trovis-modbus
requests. A reply that arrives after its request timed out stays in the
receive path and is then consumed as the answer to the *next* request.
On RTU there is no transaction ID, so the stale frame is only noticed when
its length does not fit, for example::

    read_holding_registers(9, 22): Invalid register count: expected 22, got 6

(the 6 registers are the late answer to ``read_holding_registers(0, 6)``).
Every later request is then shifted by one frame. A plain retry makes this
worse, because the retry itself consumes the next stale frame.

Countermeasures in this module
------------------------------
1. Enforce a minimum per-request timeout for this unit (``MIN_LINK_TIMEOUT``),
   regardless of what the device library asks for.
2. After a timeout or an unusable reply, *resynchronise* before retrying:
   drop the link (fresh serial/socket buffers on reconnect) and wait a short
   settle time so trailing bytes from the controller are gone.
3. Validate the length of every read result, so a misaligned frame can never
   be decoded into wrong values silently.

The module is free of Home Assistant imports so it can be unit-tested
without a Home Assistant installation.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable, Sized
from typing import Any

from modbus_connection import (
    ModbusConnectionError,
    ModbusError,
    ModbusProtocolError,
    ModbusTimeoutError,
)

_LOGGER = logging.getLogger(__name__)

# Minimum per-request timeout for the TROVIS unit, in seconds. trovis-modbus
# asks for 1 s, which is too short for FW 2.00 behind an ESPHome serial proxy.
MIN_LINK_TIMEOUT = 3.0

# Retries after a timeout (same budget as before this module existed).
READ_RETRIES = 2

# Retries after an unusable reply (stale/misaligned/corrupt frame). Kept at
# one on purpose: a reply that is unusable twice in a row after a resync is
# a real controller answer (e.g. the malformed FW 2.00 exception reply for
# unsupported coils) and must reach the coordinator so it can learn the
# address as unsupported.
PROTOCOL_RETRIES = 1

# Pause after dropping the link, in seconds. The controller was observed to
# keep sending residue for ~100 ms after a malformed reply.
RESYNC_SETTLE_SECONDS = 0.5

type LogFunc = Callable[..., None]
type SleepFunc = Callable[[float], Awaitable[Any]]


class ReadRetryModbusUnit:
    """TROVIS-local ModbusUnit proxy with resynchronising read retries.

    The Home Assistant unit may be shared, so methods are never monkeypatched
    on the Core-owned object. This proxy only affects the TROVIS consumer and
    forwards every other attribute unchanged.

    Writes are deliberately NOT retried here: repeating a write after a
    timeout could execute an action twice.
    """

    def __init__(
        self,
        unit: Any,
        device_name: str,
        *,
        log: LogFunc | None = None,
        sleep: SleepFunc = asyncio.sleep,
        min_timeout: float = MIN_LINK_TIMEOUT,
        settle_seconds: float = RESYNC_SETTLE_SECONDS,
    ) -> None:
        """Wrap ``unit`` for the controller named ``device_name``."""
        if min_timeout < 0 or settle_seconds < 0:
            raise ValueError("timeouts must be non-negative")
        self._unit = unit
        self._device_name = device_name
        self._log: LogFunc = log or _LOGGER.debug
        self._sleep = sleep
        self._min_timeout = min_timeout
        self._settle_seconds = settle_seconds
        self.resync_count = 0  # diagnostics: number of link resyncs so far

    def __getattr__(self, name: str) -> Any:
        """Forward every operation not overridden below to the shared unit."""
        return getattr(self._unit, name)

    # -- link requirements -------------------------------------------------

    def require_timeout(self, seconds: float | None) -> None:
        """Forward a timeout requirement, never below ``min_timeout``.

        ``None`` (withdraw) is replaced by the TROVIS minimum, because this
        unit always needs the longer timeout on slow links.
        """
        required = self._min_timeout if seconds is None else max(
            float(seconds), self._min_timeout
        )
        require_timeout = getattr(self._unit, "require_timeout", None)
        if callable(require_timeout):
            require_timeout(required)

    # -- reads -------------------------------------------------------------

    async def read_holding_registers(self, address: int, count: int) -> list[int]:
        """Read holding registers with resynchronising retries."""
        return await self._read_with_retry("read_holding_registers", address, count)

    async def read_coils(self, address: int, count: int) -> list[bool]:
        """Read coils with resynchronising retries."""
        return await self._read_with_retry("read_coils", address, count)

    async def _read_with_retry(
        self,
        method_name: str,
        address: int,
        count: int,
    ) -> Any:
        """Run one read; resync and retry on timeouts and unusable replies."""
        method = getattr(self._unit, method_name)
        call = f"{method_name}({address}, {count})"
        timeouts = 0
        unusable = 0

        while True:
            try:
                result = await method(address, count)
                _check_length(call, result, count)
            except ModbusTimeoutError:
                if timeouts >= READ_RETRIES:
                    self._log(
                        "%s: %s timed out on all %d attempts.",
                        self._device_name,
                        call,
                        READ_RETRIES + 1,
                    )
                    raise
                timeouts += 1
                self._log(
                    "%s: %s timed out, resync and retry %d/%d.",
                    self._device_name,
                    call,
                    timeouts,
                    READ_RETRIES,
                )
                await self._resync()
                continue
            except ModbusProtocolError as err:
                # Includes ModbusDesyncError and our own length check.
                if unusable >= PROTOCOL_RETRIES:
                    raise
                unusable += 1
                self._log(
                    "%s: %s unusable reply (%s), resync and retry %d/%d.",
                    self._device_name,
                    call,
                    err,
                    unusable,
                    PROTOCOL_RETRIES,
                )
                await self._resync()
                continue

            if timeouts or unusable:
                self._log(
                    "%s: %s succeeded after %d timeout(s) and %d unusable "
                    "reply(ies).",
                    self._device_name,
                    call,
                    timeouts,
                    unusable,
                )
            return result

    async def _resync(self) -> None:
        """Drop the link and let the line settle before the next request.

        Reconnecting gives the backend fresh receive buffers, so a late or
        trailing frame cannot be paired with the next request. Errors while
        disconnecting are irrelevant here: the next request reconnects anyway.
        """
        self.resync_count += 1
        disconnect = getattr(self._unit, "disconnect", None)
        if callable(disconnect):
            with contextlib.suppress(ModbusConnectionError, ModbusError, OSError):
                await disconnect()
        await self._sleep(self._settle_seconds)


class ReadLengthError(ModbusProtocolError):
    """A read returned a different number of values than requested."""


def _check_length(call: str, result: Any, count: int) -> None:
    """Raise ReadLengthError if ``result`` does not hold ``count`` values.

    Coil reads may legitimately be padded to full bytes by some backends,
    so for coils only *fewer* values than requested is an error.
    """
    if not isinstance(result, Sized):
        return
    length = len(result)
    if call.startswith("read_coils"):
        if length < count:
            raise ReadLengthError(
                f"{call}: expected at least {count} coils, got {length}"
            )
        return
    if length != count:
        raise ReadLengthError(
            f"{call}: expected {count} registers, got {length}"
        )


def with_read_retries(unit: Any, device_name: str, **kwargs: Any) -> Any:
    """Return a TROVIS-local unit view with resynchronising read retries."""
    if isinstance(unit, ReadRetryModbusUnit):
        return unit
    return ReadRetryModbusUnit(unit, device_name, **kwargs)
