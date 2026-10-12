"""Root actions around ``sparkring dashboard`` on Node A: its site file, its service and setup's plan line.

The serving collector (``runtime.host.dashboard``) starts no process and needs no privilege.
The actions here run as root, in the operator's commands:

- ``write_site`` turns the controller's deployment records (root only) into the site file
  (``sparkring-dashboard-site/v1``, 0644) that the collector reads.
- ``refresh_site`` is called by ``sparkring install``, ``up``, ``down``, setup and automatic
  recovery when they finish, so the page follows every change of deployment records. It writes
  the site file, then asks the running service to re-read it (``reload_service``). It never
  raises: a failure costs one line on stderr and the command carries on.
- ``after_setup`` enables and restarts ``sparkring-dashboard.service`` at the end of setup.
- ``switch`` is ``sudo sparkring dashboard enable|disable``. ``disable`` records DISABLED_PATH,
  which the unit's condition, the package's postinst and ``after_setup`` respect, so the
  dashboard stays off across reboots, upgrades and setups.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.parse

from runtime.host.dashboard import CONTAINER_RE, DEFAULT_PORT, FABRIC_PATH, SITE_PATH, SITE_SCHEMA, service_url

DISABLED_PATH = Path("/etc/sparkring/dashboard.disabled")
UNIT = "sparkring-dashboard.service"


def site_from_records(state_root: Path) -> dict:
    """The sparkring-dashboard-site/v1 document of every slot's active deployment on this controller.

    Each row: the deployment's directory name as `id`, its profile, served model name, positions
    (the whole fabric or the slot's arc), the API address and port SparkRing's own checks use,
    the display URL when one is recorded, whether its last operation stopped it, and each rank's
    model container name (``installer.containers``)."""
    from runtime.common import installer
    from runtime.host import api_endpoint, controller
    from runtime.host import placement as placements
    rows = []
    size = None
    try:
        size = controller.recorded_layout()["size"]
    except (ValueError, OSError, KeyError):
        pass
    for slot in placements.slots(state_root):
        directory = placements.recorded(state_root, slot)
        if not placements.exists(directory):
            continue
        lock = installer.read(Path(directory) / "deployment.lock.json")
        connection = installer.connection(lock)
        host = urllib.parse.urlsplit(connection["api_url"]).hostname
        positions = list(slot) if slot else list(range(size or len(lock["site"]["ranks"])))
        shown = api_endpoint.recorded(directory)
        containers = []
        try:
            for row in installer.containers(lock):
                if CONTAINER_RE.fullmatch(row["name"]):
                    containers.append({"position": positions[row["rank"]], "name": row["name"]})
        except (ValueError, KeyError, TypeError, IndexError):
            containers = []
        rows.append({"id": Path(directory).name, "profile": placements.profile_of(directory),
                     "model": connection["model"], "positions": positions, "api_host": host,
                     "api_port": connection["port"],
                     "display_url": api_endpoint.url(connection["api_url"], shown) if shown else None,
                     "stopped": placements.stopped(directory), "containers": containers})
    return {"schema": SITE_SCHEMA, "written_at": time.time(), "deployments": rows}


def write_site(output: Path | None = None) -> dict:
    """Write the site file from this controller's records; returns the document."""
    from runtime.host import controller, node
    site = site_from_records(controller.STATE)
    output = Path(output or SITE_PATH)
    node.save(output.parent, output.name, site, mode=0o644)
    return site


def _stderr(line):
    print(line, file=sys.stderr, flush=True)


def refresh_site(*, log=None, run=None) -> bool:
    """Write the site file again after a command changed deployment records and ask the service to
    re-read it; never raises. True when the file was written.

    The dashboard shows the previous records until the next refresh or
    `sudo sparkring dashboard export` when the write fails."""
    try:
        write_site()
        # The group-readable operator view follows the same changes, so a
        # member of the sparkring group reads an up-to-date status.
        from runtime.host import operator_view
        operator_view.publish()
    except Exception as error:  # noqa: BLE001 - the dashboard must never fail an installation
        (log or _stderr)(f"Dashboard: its site file was not refreshed ({type(error).__name__}: {error}); "
                         "sudo sparkring dashboard export writes it")
        return False
    reload_service(log=log, run=run)
    return True


def reload_service(*, log=None, run=None) -> bool:
    """Ask a running dashboard to re-read its host files at once (systemctl reload sends SIGHUP); a
    stopped service is left stopped. Never raises; True when systemctl accepted the request.

    The service also compares the site file's inode and time every second, so a failed request
    delays the page by at most that second."""
    if DISABLED_PATH.exists():
        return False
    try:
        _systemctl("--no-block", "try-reload-or-restart", UNIT, run=run)
        return True
    except Exception as error:  # noqa: BLE001 - the dashboard must never fail the command that wrote the file
        (log or _stderr)(f"Dashboard: {UNIT} was not asked to re-read its site file ({type(error).__name__}: "
                         f"{error}); it notices the new file within a second")
        return False


def _systemctl(*args, run=None):
    try:
        return (run or subprocess.run)(["systemctl", *args], check=True, capture_output=True, text=True, timeout=60)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(f"systemctl {' '.join(args)}: {(error.stderr or '').strip() or error}") from error
    except OSError as error:                 # no systemctl on this host
        raise RuntimeError(f"systemctl {' '.join(args)}: {error}") from error


def listen_address(api_address, management_address) -> str:
    """Where the dashboard on Node A listens: the LAN address setup records for Node A (where the
    model API of a model on Node A serves), else its management address. The collector's
    default_bind reads the same values from the fabric document."""
    return api_address or management_address or "127.0.0.1"


def plan_line(api_address, management_address) -> str:
    """The line setup's plan prints about the dashboard."""
    if DISABLED_PATH.exists():
        return "SparkRing dashboard: off (sudo sparkring dashboard enable turns it on)"
    address = listen_address(api_address, management_address)
    loopback = "" if address == "127.0.0.1" else f" and http://127.0.0.1:{DEFAULT_PORT}/ on Node A itself"
    return (f"SparkRing dashboard: http://{address}:{DEFAULT_PORT}/ on Node A{loopback}, read-only, LAN and loopback only "
            f"({UNIT}; sudo sparkring dashboard disable turns it off)")


def service_on(*, run=None) -> bool:
    """Whether the dashboard service on this host is enabled or running, and not turned off with
    `sparkring dashboard disable`. False where systemctl is absent or cannot answer."""
    if DISABLED_PATH.exists():
        return False
    for query in ("is-enabled", "is-active"):
        try:
            if (run or subprocess.run)(["systemctl", query, "--quiet", UNIT], check=False, capture_output=True,
                                       text=True, timeout=10).returncode == 0:
                return True
        except (OSError, subprocess.SubprocessError):
            return False
    return False


def page_url(*, run=None, fabric_path=None) -> str | None:
    """The address of the SparkRing dashboard page when its service is on (``service_on``), else None.

    The address follows the service's default bind rule (``dashboard.service_url``): Node A's LAN
    address from the fabric document, else loopback."""
    if not service_on(run=run):
        return None
    try:
        return service_url(Path(fabric_path or FABRIC_PATH)) + "/"
    except (OSError, ValueError, KeyError, TypeError):
        return None


def after_setup(*, log=None, run=None) -> bool:
    """Enable and restart the dashboard on Node A at the end of setup, then write its site file;
    never raises. A dashboard turned off with `sparkring dashboard disable` stays off."""
    say = log or _stderr
    started = False
    if not DISABLED_PATH.exists():
        try:
            _systemctl("enable", UNIT, run=run)
            _systemctl("restart", UNIT, run=run)
            started = True
        except Exception as error:  # noqa: BLE001 - setup has finished; the dashboard is optional
            say(f"Dashboard: {UNIT} did not start ({type(error).__name__}: {error}); "
                f"systemctl status {UNIT} shows why")
    refresh_site(log=say)
    return started


def export(output, *, run=None) -> int:
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        raise ValueError("Run sudo sparkring dashboard export on Node A: the deployment records are root's")
    site = write_site(Path(output))
    print(f"Wrote {output}: {len(site['deployments'])} deployment(s).")
    if Path(output) == Path(SITE_PATH):     # the service's own file, not a copy for another host
        reload_service(run=run)
    return 0


def switch(command, *, run=None) -> int:
    """`sudo sparkring dashboard enable|disable`: turn the service on or off for good."""
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        raise ValueError(f"Run sudo sparkring dashboard {command}")
    if command == "disable":
        DISABLED_PATH.parent.mkdir(parents=True, exist_ok=True)
        DISABLED_PATH.write_text("Written by sparkring dashboard disable: the dashboard stays off. "
                                 "sudo sparkring dashboard enable turns it on.\n", encoding="utf-8")
        _systemctl("disable", "--now", UNIT, run=run)
        print(f"Dashboard off: {UNIT} is stopped and disabled; setup and package upgrades leave it off.")
        return 0
    DISABLED_PATH.unlink(missing_ok=True)
    _systemctl("enable", "--now", UNIT, run=run)
    print(f"Dashboard on: {UNIT} serves http://NODE_A_LAN_ADDRESS:{DEFAULT_PORT}/ on Node A.")
    return 0
