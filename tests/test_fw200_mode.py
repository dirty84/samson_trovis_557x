"""Tests for operating-mode writes on firmware without HR40074 (fw200_mode.py)."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

_PATH = (
    pathlib.Path(__file__).resolve().parents[1]
    / "custom_components"
    / "samson_trovis_557x"
    / "fw200_mode.py"
)
_SPEC = importlib.util.spec_from_file_location("trovis_fw200_mode", _PATH)
assert _SPEC and _SPEC.loader
mode = importlib.util.module_from_spec(_SPEC)
sys.modules["trovis_fw200_mode"] = mode
_SPEC.loader.exec_module(mode)


class _Resolved:
    def __init__(self, address: int) -> None:
        self.address = address


class FakeUnit:
    def __init__(self, circuit: "FakeCircuit") -> None:
        self.circuit = circuit
        self.coil_writes: list[tuple[int, bool]] = []

    async def write_coil(self, address: int, value: bool) -> None:
        self.coil_writes.append((address, value))
        if self.circuit.effective:
            self.circuit.pending_autonomous = value


class FakeCircuit:
    """Rk1 with status coils; commands become visible after one refresh."""

    def __init__(self, *, effective: bool = True, has_active_mode: bool = False):
        fields = {"mode": _Resolved(105), "mode_control_autonomous": _Resolved(88)}
        if has_active_mode:
            fields["active_mode"] = _Resolved(73)
        self.resolved_fields = fields
        self.effective = effective
        self.modbus_unit = FakeUnit(self)
        self.standby = False
        self.day_active = True
        self.night_active = False
        self.automatic = True
        self.mode_control_autonomous = True
        self.pending_mode: int | None = None
        self.pending_autonomous: bool | None = None
        self.mode_writes: list[int] = []

    async def async_update(self, *, notify: bool = True) -> None:
        if self.pending_mode is not None:
            self.standby = self.pending_mode == 2
            self.day_active = self.pending_mode == 4
            self.night_active = self.pending_mode == 5
            self.mode_control_autonomous = False
        if self.pending_autonomous is not None:
            self.mode_control_autonomous = self.pending_autonomous
            self.standby = False


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    async def fake_write(circuit, field, value):
        assert field == "mode"
        circuit.mode_writes.append(int(value))
        if circuit.effective:
            circuit.pending_mode = int(value)

    async def fake_enable(unit, access_code):
        return None

    monkeypatch.setattr(mode.Component, "write", fake_write)
    import trovis_modbus.data_model as dm

    monkeypatch.setattr(dm, "async_ensure_writing_enabled", fake_enable)
    monkeypatch.setattr(mode, "VERIFY_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(mode, "VERIFY_TIMEOUT_SECONDS", 0.05)


def test_detects_missing_active_mode() -> None:
    assert mode.needs_status_coil_fallback(FakeCircuit()) is True
    assert mode.needs_status_coil_fallback(FakeCircuit(has_active_mode=True)) is False


@pytest.mark.asyncio
async def test_standby_written_once_and_verified() -> None:
    c = FakeCircuit()
    await mode.async_set_mode_via_status_coils(c, 2, access_code=1)
    assert c.mode_writes == [2]
    assert c.standby is True


@pytest.mark.asyncio
async def test_ack_without_effect_raises_and_is_not_repeated() -> None:
    c = FakeCircuit(effective=False)
    with pytest.raises(mode.Fw200ModeError):
        await mode.async_set_mode_via_status_coils(c, 2, access_code=1)
    assert c.mode_writes == [2]


@pytest.mark.asyncio
async def test_automatic_is_not_written_as_mode() -> None:
    with pytest.raises(mode.Fw200ModeError):
        await mode.async_set_mode_via_status_coils(FakeCircuit(), 1, access_code=1)


@pytest.mark.asyncio
async def test_release_writes_ownership_coil() -> None:
    c = FakeCircuit()
    await mode.async_set_mode_via_status_coils(c, 2, access_code=1)
    await mode.async_release_via_ownership_coil(c, access_code=1)
    assert c.modbus_unit.coil_writes == [(88, True)]
    assert c.mode_control_autonomous is True
