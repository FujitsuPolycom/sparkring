"""Read-only survey with fake SSH: who it reaches, how, and what the diagnosis of the result says."""
import json

import pytest

from runtime.host import cabling, survey
from runtime.host.test_cabling import FIXTURES

# Documentation addresses stand in for the Sparks' LAN addresses.
LAN = {"spark-e": "192.0.2.42", "spark-d": "192.0.2.31", "spark-b": "192.0.2.86", "spark-a": "192.0.2.15"}


def device(netdev):
    return netdev.replace("enp", "rocep").replace("enP2p", "roceP2p").rsplit("np", 1)[0]


def captured():
    document = json.loads((FIXTURES / "recabled-pairs.json").read_text(encoding="utf-8"))
    return {s["hostname"]: s for s in document["sparks"]}


def observation(spark, *, root, neighbors=()):
    """The ``survey.observe`` document of one captured Spark; LLDP only when read as root."""
    functions = []
    for row in spark["addresses"]:
        if row["ifname"] == "enP7s7":
            continue
        functions.append({"device": device(row["ifname"]), "netdev": row["ifname"], "mac": row["address"],
                          "carrier": True, "addresses": [a["local"] for a in row["addr_info"]
                                                         if a["family"] == "inet6" and a["scope"] == "link"]})
    lan = next(row["address"] for row in spark["addresses"] if row["ifname"] == "enP7s7")
    inventory = {"id": "id-" + spark["hostname"], "hostname": spark["hostname"], "architecture": "aarch64",
                 "functions": functions, "neighbors": list(neighbors), "uplink": "enP7s7",
                 "api_address": LAN[spark["hostname"]], "routes": []}
    return {"inventory": inventory, "root": root, "macs": sorted({lan} | {f["mac"] for f in functions}),
            "lldp": spark["lldp"] if root else None, "lldp_error": None if root else "unable to connect to socket"}


def link_local_neighbors(sparks, a, port_a, b, port_b):
    """Answered neighbor entries on a's port toward b's port, for functions that both have link-local addresses."""
    def functions(name, port):
        return [row for row in sparks[name]["addresses"] if row["ifname"] != "enP7s7" and cabling.port_of(row["ifname"]) == port]
    rows = []
    for local in functions(a, port_a):
        for remote in functions(b, port_b):
            address = [x["local"] for x in remote["addr_info"] if x["family"] == "inet6"]
            if address and any(x["family"] == "inet6" for x in local["addr_info"]):
                rows.append({"dst": address[0], "dev": local["ifname"], "lladdr": remote["address"], "answered": True})
    return rows


class Transport:
    """bootstrap.SSH stand-in: LAN routes sign in as the operator, whose sudo needs a password."""

    def __init__(self, documents):
        self.documents, self.logins, self.commands = documents, [], []

    def login(self, route):
        self.logins.append(route[-1]["address"])
        if route[-1]["address"] not in self.documents:
            raise ValueError(f"{route[-1]['address']} on the LAN did not accept the password")

    def command(self, route, argv, *, data=None, tty=False):
        self.commands.append((route[-1]["address"] if route else "local", argv[0]))
        if argv[0] == "sudo":
            raise RuntimeError("Bootstrap command failed: sudo: a password is required")
        return json.dumps(self.documents[route[-1]["address"]])


def scenario():
    sparks = captured()
    documents = {
        "local": observation(sparks["spark-b"], root=True),
        "root@198.51.100.2": observation(sparks["spark-a"], root=True),
        LAN["spark-e"]: observation(sparks["spark-e"], root=False,
                                       neighbors=link_local_neighbors(sparks, "spark-e", 0, "spark-d", 0)),
        LAN["spark-d"]: observation(sparks["spark-d"], root=False,
                                       neighbors=link_local_neighbors(sparks, "spark-d", 0, "spark-e", 0)
                                       + link_local_neighbors(sparks, "spark-d", 1, "spark-a", 0)),
    }
    table = {next(r["address"] for r in s["addresses"] if r["ifname"] == "enP7s7"): LAN[name] for name, s in sparks.items()}
    return documents, table


def run(documents, table, **options):
    transport = Transport(documents)
    sweeps = []

    def local_reach_run(reach_transport, argv, *, data=None, ssh=None):
        return json.dumps(documents["local"])

    here = survey.Reach("local", "this Spark")
    here.run = local_reach_run
    found = survey.survey(transport, recorded=["root@198.51.100.2"], user="operator", say=lambda line: None,
                          ssh=lambda target, argv, data=None: json.dumps(documents[target]),
                          arp=lambda interface: dict(table), sweep=sweeps.append, here=here, **options)
    return found, transport, sweeps


def test_survey_reaches_unrecorded_sparks_over_the_lan_and_diagnoses_the_loop():
    documents, table = scenario()
    found, transport, sweeps = run(documents, table)
    names = {spark["data"]["inventory"]["hostname"]: spark["reach"].label for spark in found["sparks"].values()}
    assert names == {"spark-b": "this Spark", "spark-a": "the admin network at 198.51.100.2",
                     "spark-e": "the LAN at 192.0.2.42 as operator", "spark-d": "the LAN at 192.0.2.31 as operator"}
    assert sorted(transport.logins) == ["192.0.2.31", "192.0.2.42"] and sweeps == []
    # The operator's sudo needs a password there, so the program ran unprivileged after one sudo -n attempt.
    assert transport.commands.count(("192.0.2.42", "sudo")) == 1 and ("192.0.2.42", "python3") in transport.commands
    result = cabling.diagnose(survey.records(found), found["head"])
    assert result["fix"] == ["On spark-e, swap its two cables (port 0 ↔ port 1)."]
    assert result["order_names"] == ["spark-b", "spark-a", "spark-d", "spark-e"]
    # spark-b's port 1 has no IPv6 address and spark-e's LLDP needs sudo: only spark-b sees that cable.
    assert result["notes"] == ["spark-b port 1 ↔ spark-e port 1 was seen only from spark-b "
                               "(spark-e reports nothing on port 1)"]
    assert sorted(survey.checked_lines(found)[2:]) == [
        "  spark-d: the LAN at 192.0.2.31 as operator (LLDP not readable without sudo)",
        "  spark-e: the LAN at 192.0.2.42 as operator (LLDP not readable without sudo)"]


def test_survey_without_sign_in_reads_only_recorded_sparks_and_names_the_rest():
    documents, table = scenario()
    found, transport, _ = run(documents, table, sign_in=False)
    assert transport.logins == [] and len(found["sparks"]) == 2
    result = cabling.diagnose(survey.records(found), found["head"])
    # spark-e and spark-d are known only by name from LLDP; the cable between them was not seen.
    assert result["layout"] == "incomplete"
    assert result["summary"] == ("Four Sparks are cabled in a line; spark-d port 0 and spark-e port 0 have no "
                                 "cable seen. spark-d and spark-e were not reached, so the last cable is unknown.")


def test_failed_sign_in_and_silent_admin_target_are_noted():
    documents, table = scenario()
    del documents[LAN["spark-d"]]
    del documents["root@198.51.100.2"]

    def ssh(target, argv, data=None):
        raise RuntimeError(f"{target}: ssh: connect to host 198.51.100.2 port 2222: No route to host")
    transport = Transport(documents)
    here = survey.Reach("local", "this Spark")
    here.run = lambda t, argv, data=None, ssh=None: json.dumps(documents["local"])
    found = survey.survey(transport, recorded=["root@198.51.100.2"], user="operator", say=lambda line: None, ssh=ssh,
                          arp=lambda interface: dict(table), sweep=lambda interface: None, here=here)
    assert "198.51.100.2 (recorded cluster) did not answer over the admin network" in found["notes"][0]
    assert any("Sign-in over the LAN at 192.0.2.31 as operator failed" in note for note in found["notes"])


def test_lan_sweep_runs_once_when_no_cabled_spark_is_listed():
    documents, table = scenario()
    calls = []

    def arp(interface):
        calls.append(interface)
        return dict(table) if len(calls) > 1 else {}
    transport = Transport(documents)
    here = survey.Reach("local", "this Spark")
    here.run = lambda t, argv, data=None, ssh=None: json.dumps(documents["local"])
    sweeps = []
    survey.survey(transport, recorded=[], user="operator", say=lambda line: None, arp=arp, sweep=sweeps.append, here=here)
    assert sweeps == ["enP7s7"]


def test_program_is_self_contained():
    plain = survey.program()
    compile(plain, "observe", "exec")
    assert "def prior_state" not in plain and plain.rstrip().endswith("print(json.dumps(observe(False)))")


@pytest.mark.parametrize("kind", ["local", "target"])
def test_root_reaches_do_not_fall_back_to_unprivileged_runs(kind):
    reach = survey.Reach(kind, "x", target="root@198.51.100.2")
    calls = []

    def ssh(target, argv, data=None):
        calls.append(argv[0])
        raise RuntimeError("failed")
    if kind == "local":
        reach.run = lambda t, argv, data=None, ssh=None: (calls.append(argv[0]), (_ for _ in ()).throw(RuntimeError("failed")))
    with pytest.raises(RuntimeError):
        survey.run_program(reach, None, ssh=ssh)
    assert calls == ["sudo"]


def test_cabling_command_prints_what_it_read_and_the_fix(monkeypatch, capsys, tmp_path):
    import os
    from runtime.host import controller
    documents, table = scenario()
    found, _, _ = run(documents, table)
    monkeypatch.setattr(os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(controller, "STATE", tmp_path)
    seen = {}

    def fake_survey(transport, **options):
        seen.update(options)
        return found
    monkeypatch.setattr(survey, "survey", fake_survey)
    assert cabling.main(["--ssh-user", "operator"]) == 1
    output = capsys.readouterr().out.splitlines()
    assert seen["recorded"] == [] and seen["user"] == "operator" and seen["sign_in"]
    assert output[:2] == ["Sparks read:", "  spark-b: this Spark"]
    assert "  On spark-e, swap its two cables (port 0 ↔ port 1)." in output
    assert cabling.main(["--json", "--no-sign-in"]) == 1
    document = json.loads(capsys.readouterr().out)
    assert document["schema"] == cabling.SCHEMA and not seen["sign_in"]
    assert document["order_names"] == ["spark-b", "spark-a", "spark-d", "spark-e"]


def test_cabling_command_needs_root(monkeypatch, capsys):
    import os
    monkeypatch.setattr(os, "geteuid", lambda: 1000, raising=False)
    assert cabling.main([]) == 2
    assert "Run sudo sparkring cabling" in capsys.readouterr().err
