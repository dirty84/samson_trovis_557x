"""Operating-mode writes for controllers without the effective-mode register.

Why this exists
---------------
trovis-modbus verifies an operating-mode change through ``active_mode``
(HR40074 + Rk index). TROVIS 5573 firmware 2.00 does not implement
HR40074-40077 (they end up in the learned read exclusions), so the library
cannot resolve ``active_mode`` and aborts *before* writing anything::

    AttributeError: unknown field 'active_mode'

This module provides the same two operations for such controllers:

* set an external operating mode (STANDBY / DAY / NIGHT ...) by writing the
  command register (HR40106 + 2*(Rk-1)) exactly once, then verifying the
  result through the controller's status coils (CL1001 ff.: automatic, day,
  night, standby), which FW 2.00 does implement;
* release operating-mode ownership back to the controller (AUTARK) by writing
  the ownership coil (CL90 + 2*(Rk-1)) and verifying it by readback.

Writes are never repeated after an acknowledged write; only a timed-out
write (lost response) is retried once, after re-checking the status first.
"""

from __future__ import annotations

import asyncio
import logging
from enum import IntEnum
from time import monotonic
from typing import Any

from modbus_connection import ModbusTimeoutError
from modbus_connection.model import Component

_LOGGER = logging.getLogger(__name__)

# Time the controller gets to reflect a new mode in its status coils, and the
# poll interval while waiting. Status coils follow within seconds on FW 2.00
# behind a slow serial proxy, so allow generous headroom.
VERIFY_TIMEOUT_SECONDS = 30.0
VERIFY_INTERVAL_SECONDS = 3.0
WRITE_RETRIES = 1

# OperatingMode values (trovis_modbus.enums.OperatingMode), duplicated here to
# keep the status mapping explicit and independent of enum member names.
_AUTOMATIC = 1
_STANDBY = 2
_DAY = 4
_NIGHT = 5


class Fw200ModeError(RuntimeError):
    """An operating-mode change could not be written or verified."""


def needs_status_coil_fallback(circuit: Any) -> bool:
    """Return True if the circuit cannot resolve its effective-mode register."""
    resolved = getattr(circuit, "resolved_fields", None)
    if resolved is None:
        return False
    return "active_mode" not in resolved and "mode" in resolved


def _mode_reached(circuit: Any, mode: int) -> bool | None:
    """Return whether the status coils show ``mode`` (None if unknown)."""
    standby = getattr(circuit, "standby", None)
    if mode == _STANDBY:
        return None if standby is None else bool(standby)
    if mode == _DAY:
        day = getattr(circuit, "day_active", None)
        if day is None or standby is None:
            return None
        return bool(day) and not bool(standby)
    if mode == _NIGHT:
        night = getattr(circuit, "night_active", None)
        if night is None or standby is None:
            return None
        return bool(night) and not bool(standby)
    if mode == _AUTOMATIC:
        automatic = getattr(circuit, "automatic", None)
        return None if automatic is None else bool(automatic)
    return None


async def _refresh(circuit: Any) -> None:
    """Re-read the circuit (status coils included) without notifying listeners."""
    await circuit.async_update(notify=False)


async def _wait_until(circuit: Any, check: Any, label: str) -> bool:
    """Poll the circuit until ``check()`` is True or the timeout expires."""
    started = monotonic()
    while True:
        try:
            await _refresh(circuit)
        except ModbusTimeoutError as err:
            _LOGGER.debug("%s: status refresh timed out: %s", label, err)
        else:
            if check() is True:
                return True
        if monotonic() - started >= VERIFY_TIMEOUT_SECONDS:
            return False
        await asyncio.sleep(VERIFY_INTERVAL_SECONDS)


async def async_set_mode_via_status_coils(
    circuit: Any,
    mode: IntEnum | int,
    *,
    access_code: int,
    label: str = "Rk",
) -> None:
    """Write an external operating mode and verify it via status coils."""
    from trovis_modbus.data_model import async_ensure_writing_enabled

    requested = int(mode)
    if requested not in (_STANDBY, _DAY, _NIGHT):
        raise Fw200ModeError(
            f"{label}: mode {requested} cannot be verified on this firmware; "
            "use release (automatic) instead"
        )

    unit = circuit.modbus_unit
    await async_ensure_writing_enabled(unit, access_code)

    for attempt in range(WRITE_RETRIES + 1):
        try:
            # Plain component write: no generic "Ebene=GLT" pre-write. A valid
            # external mode command switches ownership to GLT by itself.
            await Component.write(circuit, "mode", mode)
        except ModbusTimeoutError as err:
            _LOGGER.warning(
                "%s: mode command timed out (attempt %d/%d), checking status "
                "before any retry: %s",
                label,
                attempt + 1,
                WRITE_RETRIES + 1,
                err,
            )
            if await _wait_until(
                circuit, lambda: _mode_reached(circuit, requested), label
            ):
                return
            continue

        if await _wait_until(
            circuit, lambda: _mode_reached(circuit, requested), label
        ):
            _LOGGER.info("%s: operating mode %s confirmed by status coils", label, requested)
            return
        # Acknowledged but not effective: never repeat an acknowledged write.
        break

    raise Fw200ModeError(
        f"{label}: operating mode {requested} was not confirmed by the status "
        f"coils within {VERIFY_TIMEOUT_SECONDS:.0f}s"
    )


async def async_release_via_ownership_coil(
    circuit: Any,
    *,
    access_code: int,
    label: str = "Rk",
) -> None:
    """Hand operating-mode control back to the controller (AUTARK)."""
    from trovis_modbus.data_model import async_ensure_writing_enabled

    resolved = circuit.resolved_fields.get("mode_control_autonomous")
    if resolved is None:
        raise Fw200ModeError(f"{label}: ownership coil is not readable")

    unit = circuit.modbus_unit
    await async_ensure_writing_enabled(unit, access_code)

    def _released() -> bool | None:
        value = getattr(circuit, "mode_control_autonomous", None)
        return None if value is None else bool(value)

    for attempt in range(WRITE_RETRIES + 1):
        try:
            await unit.write_coil(resolved.address, True)
        except ModbusTimeoutError as err:
            _LOGGER.warning(
                "%s: release timed out (attempt %d/%d): %s",
                label,
                attempt + 1,
                WRITE_RETRIES + 1,
                err,
            )
            if await _wait_until(circuit, _released, label):
                return
            continue

        if await _wait_until(circuit, _released, label):
            _LOGGER.info("%s: operating-mode ownership released (AUTARK)", label)
            return
        break

    raise Fw200ModeError(
        f"{label}: ownership release was not confirmed within "
        f"{VERIFY_TIMEOUT_SECONDS:.0f}s"
    )
