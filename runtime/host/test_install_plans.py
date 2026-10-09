"""``sparkring install --plan`` on SIRCL and on the prepared transport stays what fixtures/install-plans.json records.

The simulated two-Spark machine of ``test_install_sircl`` plans one
installation on SIRCL ring sessions (the default) and one with
``--transport prepared``. Each result document and its printed plan are
compared, after the test's temporary directory is replaced by ``<tmp>``,
with the fixture, which revision ``generated_at`` of the fixture wrote. A
change that is meant to alter these plans rewrites the fixture in the same
commit: run this file with ``SPARKRING_WRITE_INSTALL_PLANS=1``.
"""
import json
import os
from pathlib import Path
import re

from runtime.host.test_install_sircl import install, sircl  # noqa: F401  (pytest fixture)
from runtime.host.test_install_workflow import machine, sparks  # noqa: F401  (pytest fixtures)

FIXTURE = Path(__file__).with_name("fixtures") / "install-plans.json"
SCHEMA = "sparkring-install-plans/v1"
REQUESTS = {"sircl": (), "prepared": ("--transport", "prepared")}


def normalized(text, root):
    """``text`` with the temporary directory ``root`` as ``<tmp>``, POSIX separators and no shell quotes."""
    for prefix in (str(root), str(root).replace("\\", "\\\\")):
        text = text.replace(prefix, "<tmp>")
    text = re.sub(r"<tmp>[^\s'\"]*", lambda found: found.group(0).replace("\\\\", "/").replace("\\", "/"), text)
    return re.sub(r"'(<tmp>[^']*)'", r"\1", text)


def test_the_sircl_and_prepared_install_plans_are_those_the_fixture_records(sircl, capsys, tmp_path):  # noqa: F811
    _, lock, _ = sircl
    rendered = {}
    for name, extra in REQUESTS.items():
        assert install(lock, "--plan", *extra) == 0
        out = capsys.readouterr()
        rendered[name] = {"result": json.loads(normalized(out.out, tmp_path)),
                          "printed": normalized(out.err, tmp_path).splitlines()}
    if os.environ.get("SPARKRING_WRITE_INSTALL_PLANS") == "1":
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        value = {"schema": SCHEMA, "generated_at": os.environ.get("SPARKRING_PLANS_REVISION"), "plans": rendered}
        FIXTURE.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    recorded = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert recorded["schema"] == SCHEMA
    for name in REQUESTS:
        assert rendered[name]["result"] == recorded["plans"][name]["result"], name
        assert rendered[name]["printed"] == recorded["plans"][name]["printed"], name
