import json

import pytest

from runtime.common import image_lock, installer_image, setup
from runtime.host import images

MIMO = ("mimo-v26-flash-mopd-tp2", "mimo-v26-flash-mopd-tp4")


def test_images_lists_the_default_first_and_each_image_s_release_tag(capsys):
    assert images.main([]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "dev-20261016-sparkcache-4a17f1f5198e-status038  (2026.10.5, default)"
    assert "dev-20261013-dcp2prefill-status038-sircl-tp6  (2026.10.4)" in lines
    assert "dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036  (2026.10.2)" in lines
    from runtime.common import installer
    for row in images.rows():
        heading = next(i for i, line in enumerate(lines) if line.split("  (")[0] == row["name"])
        claim = lines[heading + 1].split("; runs ", 1)[1]
        if claim == "every installer profile":
            claimed = set(installer.INSTALLABLE)
        elif claim.startswith("every installer profile except "):
            claimed = set(installer.INSTALLABLE) - set(claim.removeprefix("every installer profile except ").split(", "))
        else:
            claimed = set(claim.split(", "))
        assert claimed == set(row["profiles"]), row["tags"]
    releases = {tag: set(row["profiles"]) for row in images.rows() for tag in row["tags"]}
    assert not {"glm53-nvfp4-csf-tp6", "glm53-flash-nvfp4-spark-tp2-dcp2"} & releases["2026.10.2"]
    assert not set(installer_image.SIRCL_ONLY) & releases["2026.10.1"]
    assert lines[-1] == "Install a profile on one of them: sudo sparkring install --profile PROFILE --image NAME"


@pytest.mark.parametrize("variant", installer_image.UNLISTED)
def test_no_image_of_this_package_lists_an_unlisted_profile(capsys, variant):
    # Only a lock that names them admits them, such as the one the SparkCache variants' derived layer
    # records, so no listed image runs them.
    assert images.main(["--profile", variant, "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


@pytest.mark.parametrize("variant", installer_image.LISTED_SPARKCACHE_VARIANTS)
def test_only_the_sparkcache_image_lists_a_listed_sparkcache_variant(capsys, variant):
    # Release 2026.10.5's lock, the default, is the only published lock that names the SparkCache variants
    # its composition was built for.
    assert images.main(["--profile", variant, "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert [row["name"] for row in listed] == ["dev-20261016-sparkcache-4a17f1f5198e-status038"]
    assert listed[0]["default"] and listed[0]["tags"] == ["2026.10.5"]


@pytest.mark.parametrize("variant", installer_image.DCP_VARIANTS)
def test_an_image_that_lists_a_dcp_variant_has_the_tp2_dcp2_prefill_vllm_that_reads_csf(variant):
    # A published lock may list a decode-context-parallel variant only when it is a v3 lock whose vLLM is the
    # TP2/DCP2 prefill build, which reads the CSF checkpoint the variant serves.
    card = setup.selection(variant)
    for row in image_lock.catalog():
        if variant in image_lock.profiles_of(row["lock"]):
            assert image_lock.schema(row["lock"]) == image_lock.SCHEMA_V3, row["name"]
            assert image_lock.checkpoint_problem(row["lock"], card) is None, row["name"]
            assert image_lock.for_profile(variant, row["lock"]) == row["lock"], row["name"]


def test_an_image_that_lists_the_dcp_variant_runs_it_and_the_default_names_it(capsys, monkeypatch):
    from runtime.common.test_image_lock import CSF_PINS, sircl_block, sircl_lock
    variant = installer_image.DCP_VARIANTS[0]
    listed = sorted({*sircl_lock()["profiles"], variant})
    value = sircl_lock(name="dev-20261013-dcp2prefill-example", profiles=listed,
                       sircl=dict(sircl_block(), vllm_pins=["sparkring-kraken-bc9ea774-tp2-dcp2-prefill"]))
    # The v2 default of this package, a published v3 image that does not list the variant, the install default,
    # and the image that lists it.
    default = installer_image.default_lock()
    published = sircl_lock()
    rows = [{"name": default["name"], "path": installer_image.DEFAULT_LOCK, "lock": default, "tags": ["2026.10.1"],
             "default": True},
            {"name": published["name"], "path": "published.json", "lock": published, "tags": ["2026.10.2"],
             "default": False},
            {"name": value["name"], "path": "v3.json", "lock": value, "tags": [], "default": False}]
    monkeypatch.setattr(installer_image, "catalog", lambda: rows)
    monkeypatch.setattr(installer_image, "release_tags", lambda: {"2026.10.1": default["name"],
                                                                  "2026.10.2": published["name"]})
    assert image_lock.default() == published
    assert images.main(["--profile", variant, "--json"]) == 0
    assert [row["name"] for row in json.loads(capsys.readouterr().out)] == [value["name"]]
    assert image_lock.for_profile(variant, value) == value
    # The install default does not list it; its refusal names the image that does.
    with pytest.raises(ValueError, match=f"images that run it: {value['name']}"):
        image_lock.for_profile(variant)
    # A lock whose vLLM reads CSF but is not the TP2/DCP2 prefill build is refused even when it lists the variant.
    with pytest.raises(ValueError, match="source-pinned vLLM build"):
        image_lock.for_profile(variant, dict(value, sircl=dict(sircl_block(), vllm_pins=CSF_PINS)))


def test_images_for_a_profile_lists_only_images_that_run_it(capsys):
    assert images.main(["--profile", MIMO[0], "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["default"] and all(MIMO[0] in row["profiles"] for row in listed)
    assert "dev-20260927-mimovision-cuda1342-nccl2323-status032" in [row["name"] for row in listed]
    assert not any(name.startswith("mimo-v26-flash-rl") for row in listed for name in row["profiles"])
    assert listed[0]["transports"] == ["libsircl", "prepared", "sircl"] and listed[0]["line"] == "kraken"
    # Releases 2026.10.4's and 2026.10.2's images, the default image's parent and grandparent, also carry SIRCL;
    # the older images do not.
    assert [row["name"] for row in listed[1:3]] == ["dev-20261013-dcp2prefill-status038-sircl-tp6", "dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036"]
    assert all(row["transports"] == listed[0]["transports"] for row in listed[1:3])
    assert all(row["transports"] == ["prepared"] and row["line"] is None for row in listed[3:])
    assert not any(row["archived"] for row in listed)


def test_a_v3_image_lists_its_line_and_its_transports(capsys, monkeypatch):
    from runtime.common.test_image_lock import sircl_lock
    rows = installer_image.catalog()
    value = sircl_lock()
    rows.append({"name": value["name"], "path": "v3.json", "lock": value, "tags": [], "default": False})
    monkeypatch.setattr(installer_image, "catalog", lambda: rows)
    assert images.main([]) == 0
    lines = capsys.readouterr().out.splitlines()
    shown = lines[lines.index(value["name"]) + 1]
    assert "; transports prepared, sircl; kraken line; runs " in shown


def test_images_refuses_a_name_that_is_not_an_installer_profile(capsys):
    with pytest.raises(SystemExit):
        images.main(["--profile", "qwen"])
    assert "not an installer profile" in capsys.readouterr().err


def test_images_selects_a_profile_by_its_alias(capsys):
    assert images.main(["--profile", "glm53-flash-tp4", "--json"]) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed and all("glm53-flash-nvfp4-spark-tp4" in row["profiles"] for row in listed)
