"""Installer image lock schema v3 and the image ``sparkring install`` uses by default; offline."""
import copy
import json
from pathlib import Path

import pytest

from runtime.common import image_lock, installer_image, transport
from spark_transport.sircl.sparkring_sircl import __version__ as SIRCL_VERSION

NATIVE = "0123456789abcdef"
P2P = "fedcba9876543210"


def sircl_block(version=SIRCL_VERSION, abi=9):
    """The SIRCL layer of a v3 lock; by default of ``spark_transport/sircl``'s version, as an image built from this
    tree carries it."""
    return {"version": version, "abi_version": abi,
            "wheel": {"name": f"sparkring_sircl-{version}-py3-none-any.whl", "sha256": "1" * 64},
            "native": {"path": f"{image_lock.LIBRARY_DIRECTORY}/roce_proxy-{NATIVE}.so", "sha256": "2" * 64,
                       "source_digest": NATIVE},
            "p2p": {"path": f"{image_lock.LIBRARY_DIRECTORY}/p2p_proxy-{P2P}.so", "sha256": "3" * 64,
                    "source_digest": P2P},
            "receipt": {"path": image_lock.LAYER_RECEIPT, "sha256": "4" * 64},
            "tuning_key": {"native": NATIVE, "kernels": "a" * 16, "sircl": f"{version}/abi{abi}"},
            "vllm_pins": ["lil-image-aba309e4610c"]}


def sircl_lock(name="dev-20261009-kraken-sircl-cuda1342-nccl2323-status034", **changes):
    """A v3 lock of a kraken-line image with the SIRCL layer over the default image's contract."""
    value = {key: item for key, item in installer_image.default_lock().items() if key != "schema"}
    value.update(schema=image_lock.SCHEMA_V3, name=name, image_id="sha256:" + "5" * 64,
                 image_reference="ghcr.io/fujitsupolycom/sparkring@sha256:" + "6" * 64, line="kraken",
                 transports=["prepared", "sircl"], sircl=sircl_block(), archived=False,
                 tuning_defaults_sha256=transport.tuning_digest(transport.load_tuning()))
    value.update(changes)
    return value


@pytest.mark.parametrize("profile", installer_image.SUPPORTED)
def test_a_v3_lock_validates_for_every_profile_and_keeps_the_v2_contract(profile):
    value = sircl_lock()
    assert image_lock.validate(value, profile) is value
    view = image_lock.v2_view(value)
    assert view["schema"] == installer_image.SCHEMA and set(view) == installer_image.FIELDS[installer_image.SCHEMA]
    assert installer_image.validate(view, profile) is view
    assert image_lock.transports(value) == ("prepared", "sircl") and image_lock.sircl(value) == sircl_block()


def test_v1_and_v2_locks_carry_the_prepared_transport_only():
    default = installer_image.default_lock()
    assert image_lock.v2_view(default) is default
    assert image_lock.transports(default) == ("prepared",) and image_lock.sircl(default) is None
    assert image_lock.line(default) is None


@pytest.mark.parametrize("change, message", [
    ({"line": "stable"}, "image line"),
    ({"transports": ["sircl"]}, "also carry the prepared transport"),
    ({"transports": ["sircl", "prepared"]}, "sorted"),
    ({"sircl": None}, "records"),
    ({"transports": ["prepared"]}, "records no SIRCL layer"),
    ({"tuning_defaults_sha256": "x"}, "default tuning table"),
    ({"archived": "no"}, "archived"),
    ({"transport_profile": "other"}, "prepared RoCEnante"),
])
def test_a_v3_lock_with_a_wrong_field_is_refused(change, message):
    with pytest.raises(ValueError, match=message):
        image_lock.validate(sircl_lock(**change), "qwen38-flash-next-tp2")


@pytest.mark.parametrize("edit, message", [
    (lambda block: block["wheel"].update(name="sparkring_sircl-0.3.0-py3-none-any.whl"), "wheel"),
    (lambda block: block["native"].update(path="/opt/elsewhere/roce_proxy-0123456789abcdef.so"), "native library"),
    (lambda block: block["p2p"].update(source_digest="0" * 16), "p2p library"),
    (lambda block: block["tuning_key"].update(native="b" * 16), "tuning key"),
    (lambda block: block["tuning_key"].update(sircl="0.2.0/abi8"), "tuning key"),
    (lambda block: block.update(vllm_pins=["b", "a"]), "pinned vLLM builds"),
    (lambda block: block["receipt"].update(path="/tmp/receipt.json"), "receipt"),
    (lambda block: block.pop("vllm_pins"), "records"),
])
def test_a_sircl_layer_that_disagrees_with_itself_is_refused(edit, message):
    block = sircl_block()
    edit(block)
    with pytest.raises(ValueError, match=message):
        image_lock.validate_sircl(block)


def catalog_row(lock, *, default=False, tags=()):
    return {"name": lock["name"], "path": f"runtime/releases/{lock['name']}/installer-image.json", "lock": lock,
            "tags": list(tags), "default": default}


@pytest.fixture
def releases(monkeypatch):
    """A catalog with the default v2 image and a published and an unpublished v3 image."""
    default = installer_image.default_lock()
    published = sircl_lock()
    unpublished = sircl_lock(name="dev-20261010-kraken-sircl-cuda1342-nccl2323-status034")
    rows = [dict(catalog_row(default, default=True, tags=["2026.10.1"]), path=installer_image.DEFAULT_LOCK),
            catalog_row(published, tags=["2026.10.2"]), catalog_row(unpublished)]
    monkeypatch.setattr(installer_image, "catalog", lambda: copy.deepcopy(rows))
    monkeypatch.setattr(installer_image, "release_tags", lambda: {"2026.10.1": default["name"],
                                                                  "2026.10.2": published["name"]})
    return default, published, unpublished


def test_the_install_default_is_the_newest_published_kraken_image_with_sircl(releases):
    default, published, unpublished = releases
    assert image_lock.default() == published
    rows = image_lock.catalog()
    assert rows[0]["name"] == published["name"] and rows[0]["default"]
    assert [row["transports"] for row in rows if row["name"] == default["name"]] == [["prepared"]]
    # The default image is selected without --image; the v2 image needs its name or tag.
    assert image_lock.lock_path(published["name"]) is None
    assert str(image_lock.lock_path("2026.10.1")).endswith("installer-image.json")
    assert image_lock.lock_path(unpublished["name"]) is not None


def test_an_archived_image_is_never_the_default_but_stays_selectable(releases, monkeypatch):
    default, published, _ = releases
    rows = installer_image.catalog()
    rows[1]["lock"]["archived"] = True
    monkeypatch.setattr(installer_image, "catalog", lambda: copy.deepcopy(rows))
    assert image_lock.default() == default
    assert [row["archived"] for row in image_lock.catalog() if row["name"] == published["name"]] == [True]
    assert image_lock.lock_path(published["name"]) is not None


def without_published_sircl_images(monkeypatch):
    """Drop the release tags that publish an image whose lock carries SIRCL, as before the first such release."""
    locks = {row["name"]: row["lock"] for row in installer_image.catalog()}
    tags = {tag: name for tag, name in installer_image.release_tags().items()
            if not (image_lock.schema(locks.get(name, {})) == image_lock.SCHEMA_V3
                    and "sircl" in locks[name]["transports"])}
    monkeypatch.setattr(installer_image, "release_tags", lambda: tags)


def test_without_a_published_sircl_image_the_installer_default_is_unchanged(monkeypatch):
    without_published_sircl_images(monkeypatch)
    assert image_lock.default() == installer_image.default_lock()
    assert image_lock.for_profile("qwen38-flash-next-tp2") == installer_image.default_lock()


def test_the_installer_default_is_the_newest_published_sircl_release():
    row = image_lock.default_row()
    assert row["name"] == "dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036" and row["tags"] == ["2026.10.2"]
    value = image_lock.default()
    assert image_lock.schema(value) == image_lock.SCHEMA_V3 and "sircl" in value["transports"]
    assert value["image_reference"].startswith("ghcr.io/fujitsupolycom/sparkring@sha256:")
    assert image_lock.for_profile("qwen38-flash-next-tp2") == value
    # Compose exports keep installer_image's default, the v2 image of 2026.10.1.
    assert installer_image.default_lock()["name"] == "dev-20261004-kraken-cuda1342-nccl2323-status034"


def test_an_explicit_lock_that_does_not_list_the_profile_names_the_images_that_do(releases):
    _, published, _ = releases
    narrowed = dict(published, profiles=["qwen38-flash-next-tp2"])
    with pytest.raises(ValueError, match="images that run it: .*kraken"):
        image_lock.for_profile("glm53-flash-nvfp4-spark-tp2", narrowed)


def test_the_v3_schema_is_documented_with_every_field():
    text = (image_lock.installer_image.ROOT / "docs/development/releases.md").read_text(encoding="utf-8")
    for field in sorted(image_lock.V3_FIELDS - installer_image.FIELDS[installer_image.SCHEMA]):
        assert f"`{field}`" in text, field
    assert json.dumps(image_lock.SCHEMA_V3).strip('"') in text


# Profiles that only SIRCL ring sessions run.

TP8 = "glm53-flash-csf-tp8"
CSF = {"model_repository": "local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD",
       "model_revision": "dec48abd33efa73c3bb7c95b74eee10cad34f9be", "target_variant": None}


def with_eight_spark_profiles(**changes):
    profiles = sorted({*installer_image.default_lock()["profiles"], *installer_image.SIRCL_ONLY})
    return sircl_lock(profiles=profiles, **changes)


def test_only_a_v3_lock_whose_image_carries_sircl_lists_eight_spark_profiles(monkeypatch):
    value = with_eight_spark_profiles()
    assert image_lock.validate(value, TP8) is value and image_lock.for_profile(TP8, value) is value
    assert image_lock.sircl_only(value) == sorted(installer_image.SIRCL_ONLY)
    # The other profiles of the same lock keep validating through the v2 view.
    assert image_lock.validate(value, "qwen38-flash-next-tp2") is value
    with pytest.raises(ValueError, match="only when its image carries the SIRCL layer"):
        image_lock.validate(with_eight_spark_profiles(transports=["prepared"], sircl=None), TP8)
    v2 = dict(installer_image.default_lock(), profiles=sorted({*installer_image.default_lock()["profiles"], TP8}))
    with pytest.raises(ValueError, match="run only on SIRCL ring sessions"):
        image_lock.validate(v2, "qwen38-flash-next-tp2")
    # The published default image carries SIRCL and lists the profile; a v2 default refuses it and names that image.
    assert image_lock.for_profile(TP8) == image_lock.default()
    without_published_sircl_images(monkeypatch)
    with pytest.raises(ValueError, match=f"{TP8} runs only on SIRCL ring sessions, and image .* carries no SIRCL "
                                         "layer; images that run it: dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036"):
        image_lock.for_profile(TP8)


def test_the_csf_checkpoint_needs_an_image_whose_vllm_is_its_pinned_build():
    assert image_lock.checkpoint_problem(sircl_lock(), {"model_repository": "other/model", "model_revision": "0" * 40,
                                                        "target_variant": None}) is None
    problem = image_lock.checkpoint_problem(sircl_lock(), CSF)
    assert problem.startswith("Checkpoint local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD at "
                              "dec48abd33ef needs an image whose vLLM is the pinned build "
                              "sparkring-kraken-beta-20261007-bc9ea774")
    assert "matches lil-image-aba309e4610c" in problem
    assert "records no pinned vLLM build" in image_lock.checkpoint_problem(installer_image.default_lock(), CSF)
    pinned = sircl_lock(sircl=dict(sircl_block(), vllm_pins=["lil-image-aba309e4610c",
                                                             "sparkring-kraken-beta-20261007-bc9ea774"]))
    assert image_lock.checkpoint_problem(pinned, CSF) is None
    from spark_transport.sircl.sparkring_sircl.vllm import pins
    assert {build for builds in image_lock.CHECKPOINT_BUILDS.values() for build in builds} <= {
        build.name for build in pins.SUPPORTED}


# The libsircl layer, an optional block of a v3 lock.

SITE = "/usr/local/lib/python3.12/dist-packages/"


def libsircl_block(version="0.6.0"):
    return {"version": version, "source_tree": "b" * 40, "nccl_api_version": 22705, "fail_stop": True,
            "library": {"path": f"{image_lock.LIBSIRCL_LIBRARY_DIRECTORY}/libsircl.so.{version}", "sha256": "7" * 64},
            "plugin": {"name": "libsircl", "path": SITE + "sparkring_libsircl.py", "sha256": "8" * 64},
            "receipt": {"path": image_lock.LIBSIRCL_RECEIPT, "sha256": "9" * 64}}


def libsircl_lock(name="dev-20261010-kraken-sircl-libsircl-cuda1342-nccl2323-status034", **changes):
    """A v3 lock of a kraken-line image with the SIRCL and libsircl layers."""
    return sircl_lock(name, **{"transports": ["libsircl", "prepared", "sircl"], "libsircl": libsircl_block(),
                               **changes})


@pytest.mark.parametrize("profile", installer_image.SUPPORTED)
def test_a_v3_lock_with_the_libsircl_layer_validates_and_keeps_the_v2_contract(profile):
    value = libsircl_lock()
    assert image_lock.validate(value, profile) is value
    assert image_lock.transports(value) == ("libsircl", "prepared", "sircl")
    assert image_lock.libsircl(value) == libsircl_block() and image_lock.sircl(value) == sircl_block()
    assert set(image_lock.v2_view(value)) == installer_image.FIELDS[installer_image.SCHEMA]
    # A lock without the layer has no libsircl field, as every lock made before the layer existed.
    assert image_lock.libsircl(sircl_lock()) is None and "libsircl" not in sircl_lock()
    assert image_lock.libsircl(installer_image.default_lock()) is None


@pytest.mark.parametrize("change, message", [
    ({"libsircl": None}, "libsircl layer records"),
    ({"transports": ["prepared", "sircl"]}, "without the libsircl transport records no libsircl layer"),
    ({"transports": ["prepared", "libsircl", "sircl"]}, "sorted"),
])
def test_a_v3_lock_whose_libsircl_fields_disagree_is_refused(change, message):
    with pytest.raises(ValueError, match=message):
        image_lock.validate(libsircl_lock(**change), "qwen38-flash-next-tp2")
    without = {key: item for key, item in libsircl_lock().items() if key != "libsircl"}
    with pytest.raises(ValueError, match="records its libsircl layer"):
        image_lock.validate(without, "qwen38-flash-next-tp2")


@pytest.mark.parametrize("edit, message", [
    (lambda block: block["library"].update(path="/opt/elsewhere/libsircl.so.0.6.0"), "library is"),
    (lambda block: block.update(version="0.7.0"), "library is"),
    (lambda block: block.update(source_tree="ba5a337b"), "git tree id"),
    (lambda block: block.update(source_tree="b" * 41), "git tree id"),
    (lambda block: block.update(nccl_api_version="22705"), "NCCL API level"),
    (lambda block: block.update(fail_stop="yes"), "fail-stop mode"),
    (lambda block: block["plugin"].update(name="sircl"), "vLLM plugin libsircl"),
    (lambda block: block["receipt"].update(path="/tmp/receipt.json"), "receipt is"),
    (lambda block: block.pop("plugin"), "records"),
])
def test_a_libsircl_layer_that_disagrees_with_itself_is_refused(edit, message):
    block = libsircl_block()
    edit(block)
    with pytest.raises(ValueError, match=message):
        image_lock.validate_libsircl(block)


def test_a_libsircl_layer_names_its_source_by_git_tree_id_or_by_vendored_snapshot():
    block = libsircl_block()
    assert image_lock.libsircl_source(block) == ("source_tree", "b" * 40)
    vendored = {key: item for key, item in block.items() if key != "source_tree"} | {"snapshot": "a" * 64}
    assert image_lock.validate_libsircl(vendored) is vendored
    assert image_lock.libsircl_source(vendored) == ("snapshot", "a" * 64)
    with pytest.raises(ValueError, match="snapshot's tree digest"):
        image_lock.validate_libsircl(vendored | {"snapshot": "a3477af2"})
    # Exactly one source: both, or neither, is refused.
    for value in (block | {"snapshot": "a" * 64}, {key: item for key, item in block.items() if key != "source_tree"}):
        with pytest.raises(ValueError, match="its source: source_tree or snapshot"):
            image_lock.validate_libsircl(value)


RECORDED = (Path(__file__).resolve().parents[2] / "performance" / "records" / "images"
            / "dev-20261009-kraken-csf-sircl-libsircl-plugins-dcp-image-20261009")


@pytest.mark.parametrize("name", ["installer-image-68934e24.json", "installer-image-ddcd1ae6.json"])
def test_the_recorded_locks_of_a_layer_built_from_a_vendored_snapshot_validate_for_every_profile(name):
    value = json.loads((RECORDED / name).read_text(encoding="utf-8"))
    assert image_lock.libsircl_source(image_lock.libsircl(value)) == (
        "snapshot", "a3477af2ba16bbdb951b88b25c67a29402c90abc0158a1f95d82723ff6c41302")
    for profile in value["profiles"]:
        assert image_lock.validate(value, profile)


CSF_PINS = ["lil-image-aba309e4610c", "sparkring-kraken-beta-20261007-bc9ea774"]
GLM_PROFILES = ("glm53-flash-nvfp4-spark-tp2", "glm53-flash-nvfp4-spark-tp4")


@pytest.mark.parametrize("profile", GLM_PROFILES)
def test_the_glm_profiles_prefer_the_csf_checkpoint_only_on_an_image_that_reads_it(profile, monkeypatch):
    pinned = sircl_lock(sircl=dict(sircl_block(), vllm_pins=CSF_PINS))
    assert image_lock.preferred_checkpoint(pinned, profile) == "csf"
    # The default image, a SIRCL image of another vLLM build and the lock's v2 view keep the default checkpoint.
    for value in (installer_image.default_lock(), sircl_lock(), image_lock.v2_view(pinned)):
        assert image_lock.preferred_checkpoint(value, profile) is None
    from runtime.common import setup
    card = setup.selection(profile, "csf")
    assert {key: card[key] for key in CSF if key != "target_variant"} == {
        key: value for key, value in CSF.items() if key != "target_variant"}
    assert image_lock.checkpoint_problem(pinned, card) is None
    assert "(--checkpoint csf)" in image_lock.checkpoint_problem(installer_image.default_lock(), card)
    # An installation of the CSF checkpoint passed the installer's checks, so its plans state no status.
    assert image_lock.checkpoint_notice(pinned, profile, card, preferred=True) is None
    # A checkpoint with a recorded status: the plan states it, and how to install the default when the image
    # chose it.
    key = f"{card['model_repository']}@{card['model_revision']}"
    monkeypatch.setitem(image_lock.CHECKPOINT_STATUS, key, "research-only: no installation of it has passed the "
                                                           "installer's checks")
    preferred = image_lock.checkpoint_notice(pinned, profile, card, preferred=True)
    assert preferred.startswith("Checkpoint csf (local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD at "
                                "dec48abd33ef) is research-only: no installation of it has passed")
    assert f"{profile} installs it without --checkpoint" in preferred and "--checkpoint nvfp4-spark" in preferred
    explicit = image_lock.checkpoint_notice(pinned, profile, card, preferred=False)
    assert explicit.startswith("Checkpoint csf") and "without --checkpoint" not in explicit
    assert image_lock.checkpoint_notice(pinned, profile, setup.selection(profile), preferred=False) is None
    assert set(image_lock.CHECKPOINT_STATUS) <= set(image_lock.CHECKPOINT_BUILDS)


def test_a_preferred_checkpoint_is_one_that_only_some_vllm_builds_read():
    from runtime.common import profiles, toolchain_profiles, setup
    preferring = []
    pinned = sircl_lock(sircl=dict(sircl_block(), vllm_pins=CSF_PINS))
    for profile in sorted({*installer_image.SUPPORTED, *installer_image.SIRCL_ONLY}):
        card = setup.selection(profile)
        name = toolchain_profiles.preferred_checkpoint(profiles.read_json(profiles.local_path(card["configuration"])))
        if name is None:
            assert image_lock.preferred_checkpoint(pinned, profile) is None
            continue
        preferring.append(profile)
        preferred = setup.selection(profile, name)
        # Otherwise every image would install it and it would be the profile's default.
        assert f"{preferred['model_repository']}@{preferred['model_revision']}" in image_lock.CHECKPOINT_BUILDS
        assert image_lock.checkpoint_problem(installer_image.default_lock(), card) is None
    assert preferring == list(GLM_PROFILES)


# vLLM general plugins a derived layer added, an optional block of a v3 lock.

PLUGINS = {"glm53full_speedups": "1.1.0", "glm_dsa_indexer_split": "1.1.0"}


def test_a_v3_lock_lists_its_added_plugins_and_v1_v2_locks_list_none():
    value = libsircl_lock(vllm_plugins=dict(PLUGINS))
    assert image_lock.validate(value, "qwen38-flash-next-tp2") is value
    assert image_lock.vllm_plugins(value) == PLUGINS and set(image_lock.v2_view(value)) == installer_image.FIELDS[
        installer_image.SCHEMA]
    assert image_lock.vllm_plugins(sircl_lock()) == {} and "vllm_plugins" not in sircl_lock()
    assert image_lock.vllm_plugins(installer_image.default_lock()) == {}


@pytest.mark.parametrize("plugins", [{}, {"sircl": "0.3.0"}, {"b12x_loader": "1.0.0"}, {"glm53full_speedups": "1.1"},
                                     {"Glm-Split": "1.0.0"}, ["glm53full_speedups"]])
def test_a_v3_lock_whose_added_plugins_are_malformed_or_built_in_is_refused(plugins):
    with pytest.raises(ValueError, match="added vLLM plugins"):
        image_lock.validate(sircl_lock(vllm_plugins=plugins), "qwen38-flash-next-tp2")


def test_a_profile_that_loads_added_plugins_runs_only_on_a_lock_that_lists_them(monkeypatch):
    environment = {"VLLM_PLUGINS": "b12x_loader,sparkring_status,sircl,glm_dsa_indexer_split,glm53full_speedups"}
    monkeypatch.setattr(installer_image, "profile_environment", lambda profile: environment)
    assert image_lock.required_plugins(TP8) == ["glm_dsa_indexer_split", "glm53full_speedups"]
    carried = with_eight_spark_profiles(vllm_plugins=dict(PLUGINS))
    assert image_lock.plugin_problem(carried, TP8) is None and image_lock.for_profile(TP8, carried) is carried
    partial = with_eight_spark_profiles(vllm_plugins={"glm_dsa_indexer_split": "1.1.0"})
    with pytest.raises(ValueError, match=f"{TP8} loads the vLLM plugins glm53full_speedups .* its lock lists "
                                         "glm_dsa_indexer_split 1.1.0"):
        image_lock.for_profile(TP8, partial)
    with pytest.raises(ValueError, match="glm_dsa_indexer_split, glm53full_speedups .* no added vLLM plugin"):
        image_lock.for_profile(TP8, with_eight_spark_profiles())
    # The published default image lists both plugins; a v1 or v2 lock lists no added plugin and is refused.
    assert image_lock.for_profile("qwen38-flash-next-tp2") == image_lock.default()
    without_published_sircl_images(monkeypatch)
    with pytest.raises(ValueError, match="qwen38-flash-next-tp2 loads the vLLM plugins"):
        image_lock.for_profile("qwen38-flash-next-tp2")


def test_profiles_that_load_only_built_in_plugins_need_no_added_plugin():
    for profile in (*installer_image.SUPPORTED, *installer_image.SIRCL_ONLY):
        if not image_lock.required_plugins(profile):
            assert image_lock.plugin_problem(sircl_lock(), profile) is None


def test_the_glm53_tp8_profile_runs_only_on_an_image_that_carries_the_glm53_plugin_layer():
    from runtime.images import derive_glm53_plugins
    profile = "glm53-nvfp4-tp8"
    assert image_lock.required_plugins(profile) == ["glm_dsa_indexer_split", "glm53full_speedups",
                                                    "glm_dcp_decode_comm"]
    assert derive_glm53_plugins.PLUGINS == {**PLUGINS, "glm_dcp_decode_comm": "2.0.1"}
    with pytest.raises(ValueError, match=f"{profile} loads the vLLM plugins glm_dsa_indexer_split, "
                                         "glm53full_speedups, glm_dcp_decode_comm"):
        image_lock.for_profile(profile, with_eight_spark_profiles())
    # A layer with only the first two plugins (image af06e272) lacks the DCP decode plugin the profile loads.
    with pytest.raises(ValueError, match=f"{profile} loads the vLLM plugins glm_dcp_decode_comm .* its lock lists "
                                         "glm53full_speedups 1.1.0, glm_dsa_indexer_split 1.1.0"):
        image_lock.for_profile(profile, with_eight_spark_profiles(vllm_plugins=dict(PLUGINS)))
    carried = with_eight_spark_profiles(vllm_plugins=dict(derive_glm53_plugins.PLUGINS))
    assert image_lock.for_profile(profile, carried) is carried
