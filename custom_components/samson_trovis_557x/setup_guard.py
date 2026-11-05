"""Decide whether a TROVIS config entry may finish its setup.

Kept free of Home Assistant imports so it can be unit-tested directly.

Background
----------
Several entity groups are created only when the hydronic configuration
(Anlagenkennziffer / system code) is known at platform setup, e.g. the
COx-F12 control parameters (Kp, Tn, Tv, Ty) and the 3-point switch.
``Trovis557x.configuration_definition`` derives from ``info.system_code``,
which is read during the first refresh.

If the entry is (re)loaded while the Modbus link is down - typically by a
self-healing automation that calls ``homeassistant.reload_config_entry``
during an ESPHome serial-proxy outage - the first refresh fails, the system
code stays ``None`` and those entities are silently skipped for the whole
lifetime of the entry. They only reappear after the next reload with a
working link, which may be hours later.

Failing the setup with ``ConfigEntryNotReady`` in that case lets Home
Assistant retry with its built-in backoff until the controller answers,
instead of finishing with a reduced entity set. Single optional blocks that
cannot be read (TROVIS 5573-1, FW 2.00) do not affect the system code, so
the tolerant first refresh introduced for those controllers keeps working.
"""

from __future__ import annotations

from typing import Any


def controller_identity_known(device: Any) -> bool:
    """Return True once the controller's system code has been read.

    A system code that was read but is unknown to ``trovis_modbus`` still
    counts as known: the entry must load (with the generic entity set) rather
    than retry forever.
    """
    info = getattr(device, "info", None)
    if info is None:
        return False
    return getattr(info, "system_code", None) is not None


def setup_not_ready_reason(device: Any, *, last_update_success: bool) -> str | None:
    """Return a human-readable reason to retry setup, or None to proceed.

    Args:
        device: The ``Trovis557x`` instance after the first refresh.
        last_update_success: ``coordinator.last_update_success`` of that refresh.
    """
    if controller_identity_known(device):
        return None
    if last_update_success:
        # Poll succeeded but the identity block was excluded or empty: retrying
        # would not change anything, so let the entry load as before.
        return None
    return (
        "TROVIS controller not reachable during setup (system code unknown); "
        "retrying so that configuration-dependent entities (Kp/Tn/Ty, "
        "3-point mode) are created"
    )
