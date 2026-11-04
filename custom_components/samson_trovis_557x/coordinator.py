"""DataUpdateCoordinator that polls the Trovis controller."""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import timedelta
from time import monotonic

from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.update_coordinator import (
    DataUpdateCoordinator,
    UpdateFailed,
)
from modbus_connection import (
    ModbusConnectionError,
    ModbusError,
    ModbusExceptionError,
    ModbusTimeoutError,
)
from trovis_modbus import DEFAULT_WRITE_ACCESS_CODE, Trovis557x

from .const import (
    CONF_ACCESS_CODE,
    CONF_DETECTED_SENSORS,
    CONF_EXCLUDED_COILS,
    CONF_EXCLUDED_REGISTERS,
    DOMAIN,
    SCAN_INTERVAL,
)
from .modbus_link import DEGRADED_RETRY_SECONDS, LinkGrace
from .read_exclusions import format_address_list, parse_address_list

_LOGGER = logging.getLogger(__name__)
_DEV_LOGGER = logging.getLogger("TROVIS dev log")

# Modbus exception code 2: the controller does not implement the address.
# Older firmware (e.g. TROVIS 5573 V2.00) lacks registers that the model
# profile expects, so a single unknown address must not fail the full read.
_ILLEGAL_DATA_ADDRESS = 2

# Address space of a refused block -> (option key, single-address reader name).
# Only spaces that the read-exclusion options cover are handled.
_LEARNABLE_SPACES: dict[str, tuple[str, str]] = {
    "holding": (CONF_EXCLUDED_REGISTERS, "read_holding_registers"),
    "coil": (CONF_EXCLUDED_COILS, "read_coils"),
}

# Upper bound for addresses auto-excluded per space. Prevents a misbehaving
# link (wrong unit ID, foreign device) from silently excluding everything.
_MAX_AUTO_EXCLUDED_PER_SPACE = 64

# TROVIS 5573 FW 2.00 answers unsupported coils with a malformed exception
# frame (wrong byte after 0x81, trailing junk). The backend then raises a
# protocol error without block metadata, only the call prefix, e.g.
# "read_coils(401, 13): ...". Map that prefix back to the refused block.
_CALL_PREFIX = re.compile(
    r"^(read_holding_registers|read_coils)\((\d+), (\d+)\)"
)
# After a malformed reply the TROVIS keeps sending junk bytes for a while
# (observed: >200 bytes, ~100 ms). Wait before each probe so the residue
# does not corrupt the next answer and wrongly exclude a valid address.
_GARBLED_SETTLE_SECONDS = 0.3

_READER_TO_SPACE = {
    "read_holding_registers": "holding",
    "read_coils": "coil",
}

type TrovisConfigEntry = ConfigEntry["TrovisCoordinator"]


class TrovisCoordinator(DataUpdateCoordinator[Trovis557x]):
    """Poll a TROVIS controller through its Modbus unit."""

    def __init__(
        self,
        hass: HomeAssistant,
        entry: TrovisConfigEntry,
        device: Trovis557x,
        *,
        developer_mode: bool = False,
    ) -> None:
        """Initialize the TROVIS coordinator."""
        super().__init__(
            hass,
            _LOGGER,
            name=DOMAIN,
            config_entry=entry,
            update_interval=SCAN_INTERVAL,
        )

        self.device = device
        self._developer_mode = developer_mode
        self._sensor_redetect_done = False
        # Short link outages (ESPHome serial proxy) keep the last good values
        # instead of turning every entity unavailable; see LinkGrace.
        self._grace = LinkGrace()

        if developer_mode:
            _DEV_LOGGER.setLevel(logging.INFO)

    @property
    def access_code(self) -> int:
        """Return the configured TROVIS write access code."""
        return int(
            self.config_entry.data.get(
                CONF_ACCESS_CODE,
                DEFAULT_WRITE_ACCESS_CODE,
            )
        )

    async def _async_update_data(self) -> Trovis557x:
        """Refresh all TROVIS data."""
        started = monotonic()

        try:
            await self.device.async_update()
        except ModbusExceptionError as err:
            # Unknown address on this firmware: learn it, persist it as a
            # read exclusion and reload so the next poll skips it.
            if await self._async_learn_unsupported_addresses(err):
                raise UpdateFailed(
                    "Unsupported Modbus addresses excluded automatically; "
                    "reloading TROVIS integration"
                ) from err
            raise UpdateFailed(f"Error communicating with Trovis: {err}") from err
        except (ModbusConnectionError, ModbusTimeoutError) as err:
            # Link problems are never treated as unsupported addresses.
            now = monotonic()
            if self._grace.bridge(now):
                # Keep the last good values and retry sooner. Values may be
                # partly refreshed by the failed poll; all of them are real
                # controller readings.
                self.update_interval = timedelta(seconds=DEGRADED_RETRY_SECONDS)
                _LOGGER.warning(
                    "%s: link error (%s); keeping last values (%.0fs old, "
                    "failure %d), retry in %.0fs",
                    self.config_entry.title,
                    err,
                    self._grace.age(now),
                    self._grace.failures,
                    DEGRADED_RETRY_SECONDS,
                )
                return self.device
            self.update_interval = SCAN_INTERVAL
            raise UpdateFailed(f"Error communicating with Trovis: {err}") from err
        except ModbusError as err:
            if await self._async_learn_from_garbled_block(err):
                raise UpdateFailed(
                    "Unsupported Modbus addresses excluded automatically; "
                    "reloading TROVIS integration"
                ) from err
            if self._developer_mode:
                _DEV_LOGGER.info(
                    "%s: Full read failed after %.2fs.",
                    self.config_entry.title,
                    monotonic() - started,
                )

            raise UpdateFailed(f"Error communicating with Trovis: {err}") from err

        if self._developer_mode:
            _DEV_LOGGER.info(
                "%s: Full read took %.2fs.",
                self.config_entry.title,
                monotonic() - started,
            )

        ended = self._grace.success(monotonic())
        if ended:
            _LOGGER.info(
                "%s: link recovered after %d failed poll(s)",
                self.config_entry.title,
                ended,
            )
        self.update_interval = SCAN_INTERVAL
        self._redetect_sensors_once()
        return self.device

    def _redetect_sensors_once(self) -> None:
        """Persist physical sensors if the entry was stored without any.

        Entries created while the link was unstable can carry an empty sensor
        list: the setup probe failed and skipped sensor discovery. The regular
        poll reads the same sensor block (with learned exclusions applied), so
        after a successful poll the valid sensor values tell which inputs are
        wired. Store them once and reload so the measurement entities are
        created. Runs at most once per coordinator lifetime; no extra Modbus
        traffic, no reload loop (the stored list is non-empty afterwards).
        """
        if self._sensor_redetect_done:
            return
        self._sensor_redetect_done = True

        entry = self.config_entry
        stored = set(entry.data.get(CONF_DETECTED_SENSORS, ()))
        stored.update(entry.options.get(CONF_DETECTED_SENSORS, ()))
        if stored:
            return

        try:
            names = self.device.sensors.detected_sensor_names
            model_definition = self.device.model_definition
            found = sorted(
                {name for name in names if model_definition.supports_sensor(name)}
            )
        except (AttributeError, TypeError, ValueError) as err:
            _LOGGER.warning(
                "%s: Sensor re-detection skipped: %s", entry.title, err
            )
            return

        if not found:
            _LOGGER.info(
                "%s: Sensor re-detection found no physical sensors.", entry.title
            )
            return

        _LOGGER.warning(
            "%s: Physical sensors detected after setup (%s); storing them and "
            "reloading to create the measurement entities.",
            entry.title,
            ", ".join(found),
        )
        if self.hass.config_entries.async_update_entry(
            entry,
            data={**entry.data, CONF_DETECTED_SENSORS: found},
        ):
            self.hass.config_entries.async_schedule_reload(entry.entry_id)

    async def _async_learn_unsupported_addresses(
        self,
        err: ModbusExceptionError,
    ) -> bool:
        """Handle a clean "illegal data address" exception on a block read.

        Returns True if new exclusions were stored and a reload was scheduled.
        """
        block = getattr(err, "block", None)
        if block is None or err.exception_code != _ILLEGAL_DATA_ADDRESS:
            return False
        return await self._async_learn_block(
            block.space,
            block.address,
            block.count,
        )

    async def _async_learn_from_garbled_block(self, err: ModbusError) -> bool:
        """Handle a malformed reply (FW 2.00 coil exception) on a block read.

        The backend only reports the call prefix, so the block is recovered
        from the error message. Single addresses must fail twice before they
        are excluded, so line noise alone cannot exclude a valid address.
        """
        match = _CALL_PREFIX.match(str(err))
        if match is None:
            return False
        return await self._async_learn_block(
            _READER_TO_SPACE[match.group(1)],
            int(match.group(2)),
            int(match.group(3)),
            garbled=True,
        )

    async def _async_learn_block(
        self,
        space: str,
        address: int,
        count: int,
        *,
        garbled: bool = False,
    ) -> bool:
        """Exclude the refused addresses of one block and schedule a reload."""
        if space not in _LEARNABLE_SPACES or count < 1:
            return False

        option_key, reader_name = _LEARNABLE_SPACES[space]
        block_addresses = range(address, address + count)

        # Narrow a multi-address block down to the addresses that are really
        # missing, so valid neighbours in the same block keep being read.
        refused = await self._async_probe_single_addresses(
            reader_name,
            block_addresses,
            garbled=garbled,
        )
        if refused is None:
            # Link problem while probing: do not exclude anything.
            return False
        if not refused:
            if garbled:
                # Every address answered on its own: the block failure was
                # most likely transient, so exclude nothing.
                return False
            # Clean exception for the block, but every single address
            # answered: fall back to excluding the whole block.
            refused = set(block_addresses)

        entry = self.config_entry
        try:
            current = parse_address_list(str(entry.options.get(option_key, "") or ""))
        except ValueError:
            current = frozenset()

        merged = set(current) | refused
        if merged == set(current):
            # Already excluded but still refused: nothing new to learn, so
            # do not reload in a loop.
            return False
        if len(merged) > _MAX_AUTO_EXCLUDED_PER_SPACE:
            _LOGGER.error(
                "%s: Controller refused %s %s; not auto-excluding because the "
                "limit of %d excluded %s addresses would be exceeded. Check "
                "unit ID and connection",
                entry.title,
                space,
                format_address_list(refused),
                _MAX_AUTO_EXCLUDED_PER_SPACE,
                space,
            )
            return False

        value = format_address_list(merged)
        _LOGGER.warning(
            "%s: Controller does not support %s address(es) %s (%s, likely "
            "older firmware). Excluding them from reads; %s is now '%s'",
            entry.title,
            space,
            format_address_list(refused),
            "malformed exception reply" if garbled else "Modbus exception 2",
            option_key,
            value,
        )

        options = dict(entry.options)
        options[option_key] = value
        if self.hass.config_entries.async_update_entry(entry, options=options):
            self.hass.config_entries.async_schedule_reload(entry.entry_id)
        return True

    async def _async_probe_single_addresses(
        self,
        reader_name: str,
        addresses: range,
        *,
        garbled: bool = False,
    ) -> set[int] | None:
        """Return the addresses the controller refuses when read one by one.

        Returns None if the link failed while probing. With ``garbled`` set,
        a malformed reply counts as refusal only if it repeats on a retry.
        """
        unit = getattr(self.device, "_unit", None)
        reader = getattr(unit, reader_name, None) if unit is not None else None
        if reader is None:
            # No direct reader: the whole block is the best answer, but only
            # for clean exceptions; never guess on malformed replies.
            return None if garbled else set(addresses)
        if len(addresses) == 1 and not garbled:
            # Single-address block with a clean exception: already decided.
            return set(addresses)

        refused: set[int] = set()
        for address in addresses:
            attempts = 2 if garbled else 1
            for attempt in range(attempts):
                if garbled:
                    await asyncio.sleep(_GARBLED_SETTLE_SECONDS)
                try:
                    await reader(address, 1)
                except ModbusExceptionError as err:
                    if err.exception_code == _ILLEGAL_DATA_ADDRESS:
                        refused.add(address)
                    else:
                        # Any other exception: do not guess, keep the address.
                        _LOGGER.debug(
                            "Probe of address %d returned exception %s",
                            address,
                            err.exception_code,
                        )
                    break
                except (ModbusConnectionError, ModbusTimeoutError) as err:
                    # Link problem during probing: abort, exclude nothing new.
                    _LOGGER.debug(
                        "Probing aborted at address %d: %s", address, err
                    )
                    return None
                except ModbusError as err:
                    if not garbled:
                        _LOGGER.debug(
                            "Probing aborted at address %d: %s", address, err
                        )
                        return None
                    if attempt == attempts - 1:
                        # Malformed reply twice in a row: unsupported.
                        refused.add(address)
                else:
                    break
        return refused
