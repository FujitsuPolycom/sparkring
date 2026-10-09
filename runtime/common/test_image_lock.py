"""Installer image lock schema v3 and the image ``sparkring install`` uses by default; offline."""
import copy
import json

import pytest

from runtime.common import image_lock, installer_image, transport

NATIVE = "0123456789abcdef"
P2P = "fedcba9876543210"


def sircl_block(version="0.2.0", abi=9):
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


def test_without_a_published_sircl_image_the_installer_default_is_unchanged():
    assert image_lock.default() == installer_image.default_lock()
    assert image_lock.for_profile("qwen38-flash-next-tp2") == installer_image.default_lock()


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


def test_only_a_v3_lock_whose_image_carries_sircl_lists_eight_spark_profiles():
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
    with pytest.raises(ValueError, match=f"{TP8} runs only on SIRCL ring sessions, and image .* carries no SIRCL "
                                         "layer; images that run it: none in this package"):
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
    return {"version": version, "snapshot": "b" * 64, "nccl_api_version": 22705,
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
    (lambda block: block.update(snapshot="ba5a337b"), "tree digest"),
    (lambda block: block.update(nccl_api_version="22705"), "NCCL API level"),
    (lambda block: block["plugin"].update(name="sircl"), "vLLM plugin libsircl"),
    (lambda block: block["receipt"].update(path="/tmp/receipt.json"), "receipt is"),
    (lambda block: block.pop("plugin"), "records"),
])
def test_a_libsircl_layer_that_disagrees_with_itself_is_refused(edit, message):
    block = libsircl_block()
    edit(block)
    with pytest.raises(ValueError, match=message):
        image_lock.validate_libsircl(block)
