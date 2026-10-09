"""The derived Qwen checkpoint's manifest, selection, storage figure and rendered containers."""
import copy
import hashlib
import json

import pytest

from runtime.common import derived_checkpoint, installer, mxfp8_attention, toolchain_profiles, setup
from runtime.common.test_compose_installer import install_site

NAME = "qad-step5500-mxfp8-attention"
PROFILES = ("qwen38-flash-next-tp2", "qwen38-flash-next-qad-tp4")
BASE = "60215d26cf5e42c2db6128774032d57fc62678da"
DONOR = "629bc3218833a38b475b719f34aa571666f4a03e"
REPOSITORY = "local-inference-lab/Qwen3.8-Flash-Next-NVFP4"
RECIPE_FILES = ["config.json", "derivation.json", "hf_quant_config.json", "model-00017-of-00041.safetensors",
                "model-00018-of-00041.safetensors", "model.safetensors.index.json"]


def inputs(profile=PROFILES[0]):
    card = setup.selection(profile, NAME)
    configuration = installer.read(installer.ROOT / card["configuration"])
    manifest = derived_checkpoint.load(card)
    base = installer.checkpoint_pins(setup.selection(profile))
    donor = installer.checkpoint_pins(setup.selection(profile, "qad-step-4000"))
    return card, configuration, manifest, base, donor


def test_the_selection_names_the_base_and_the_manifest_pins_every_derived_file():
    for profile in PROFILES:
        card, configuration, manifest, base, donor = inputs(profile)
        # The installer acquires the base with the checkpoint search, adoption, downloads and fabric copies.
        assert (card["target_variant"], card["model_repository"], card["model_revision"]) == (NAME, REPOSITORY, BASE)
        assert installer.checkpoint_pins(card) == base
        assert installer.checkpoint_directory("ring", card) == f"/srv/sparkring/ring/checkpoints/{REPOSITORY.replace('/', '--')}/{BASE}"
        entry = configuration["checkpoints"][NAME]
        assert entry["derived"] == {"base": "qad-step5500-ple1000", "donor": "qad-step-4000"}
        assert (manifest["repository"], manifest["revision"]) == (entry["model"]["repository"], entry["model"]["revision"])
        assert manifest["revision"] == derived_checkpoint.identity(manifest["base"], manifest["donor"], manifest["recipe"])
        assert manifest["donor"] == {"repository": REPOSITORY, "revision": DONOR,
                                     "files": ["model-00035-of-00036.safetensors", "model.safetensors.index.json"]}
        recipe = installer.ROOT / manifest["recipe"]["path"]
        assert hashlib.sha256(recipe.read_bytes()).hexdigest() == manifest["recipe"]["sha256"]
        assert recipe.resolve() == __import__("pathlib").Path(mxfp8_attention.__file__).resolve()
        # 48 files of the base stay unchanged; the recipe writes six.
        assert sorted(derived_checkpoint.files_of(manifest, "recipe")) == RECIPE_FILES
        kept = derived_checkpoint.files_of(manifest, "base")
        assert len(kept) == 48 and len(manifest["files"]) == 54
        for name, size in kept.items():
            assert (size, manifest["files"][name]["sha256"]) == (base["files"][name]["size"], base["files"][name]["sha256"])
        assert sum(item["size"] for item in manifest["files"].values()) == 107_570_132_364
        assert sum(derived_checkpoint.files_of(manifest, "recipe").values()) == 6_021_571_547
        # The record is the derived identity with the transform; derivation.json pins its canonical bytes.
        record = derived_checkpoint.record(manifest)
        assert record["checkpoint"] == {"repository": manifest["repository"], "revision": manifest["revision"]}
        assert record["modules"] == {"count": 240, "sha256": "24bac2b336bc7755dd5e055f09289797284ad9a2b548122886d8bc73051a0504"}
        written = derived_checkpoint.record_bytes(record)
        assert manifest["files"]["derivation.json"]["sha256"] == hashlib.sha256(written).hexdigest()
        # The recipe writes the record with the same canonical form.
        assert mxfp8_attention.record_bytes(record) == written


def mutate(manifest, path, value):
    result = copy.deepcopy(manifest)
    target = result
    for key in path[:-1]:
        target = target[key]
    if value is KeyError:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return result


@pytest.mark.parametrize("path, value, message", [
    (("schema",), "sparkring-checkpoint-pins/v1", "expected sparkring-derived-checkpoint/v1"),
    (("revision",), "0" * 40, "names another checkpoint"),
    (("base", "revision"), DONOR, "base differs"),
    (("donor", "files"), ["model.safetensors.index.json", "model-00035-of-00036.safetensors"], "donor files"),
    (("donor", "files"), ["README.md", "model.safetensors.index.json"], "donor files"),
    (("recipe", "sha256"), "0" * 64, "differs from the pinned recipe"),
    (("transform", "modules", "count"), 239, "derivation.json differs from the record"),
    (("files", "vocab.json", "sha256"), "0" * 64, "vocab.json differs from the base's pins"),
    (("files", "vocab.json"), KeyError, "lacks base files: vocab.json"),
    (("files", "config.json", "sha256"), "0" * 64, "config.json or the index differs"),
    (("files", "README.md"), {"origin": "recipe", "size": 1, "sha256": "0" * 64}, "README.md is neither"),
    (("index",), "vocab.json", "index must name a recipe file"),
])
def test_manifest_refusals_name_the_field_or_file(path, value, message):
    card, configuration, manifest, base, donor = inputs()
    model = configuration["checkpoints"][NAME]["model"]
    changed = mutate(manifest, path, value)
    with pytest.raises(ValueError, match=message):
        derived_checkpoint.check(changed, model, base, donor)


def test_the_identity_covers_the_base_the_donor_files_and_the_recipe():
    _, _, manifest, _, _ = inputs()
    revision = derived_checkpoint.identity(manifest["base"], manifest["donor"], manifest["recipe"])
    for key, field, value in (("base", "revision", DONOR), ("donor", "files", ["model.safetensors.index.json"]),
                              ("recipe", "sha256", "1" * 64), ("recipe", "path", "runtime/common/other.py")):
        changed = mutate(manifest, (key, field), value)
        assert derived_checkpoint.identity(changed["base"], changed["donor"], changed["recipe"]) != revision


def test_derived_and_donor_directories_lie_beside_the_base_checkpoint_directory():
    _, _, manifest, _, _ = inputs()
    root = "/srv/sparkring/ring/checkpoints"
    base = f"{root}/{REPOSITORY.replace('/', '--')}/{BASE}"
    assert derived_checkpoint.directory(base, manifest) == (
        f"{root}/sparkring-derived--Qwen3.8-Flash-Next-NVFP4-QAD5500-MXFP8-Attention/{manifest['revision']}")
    assert derived_checkpoint.directory(base, manifest["donor"]) == f"{root}/{REPOSITORY.replace('/', '--')}/{DONOR}"
    for named in ("/home/operator/models/qwen", "/srv/sparkring/ring/models/" + BASE, "/" + BASE):
        with pytest.raises(ValueError, match="never beside a named copy"):
            derived_checkpoint.directory(named, manifest)


def test_each_view_of_a_derived_selection_has_its_own_contract():
    card, configuration, manifest, base, donor = inputs()
    derived = derived_checkpoint.view(card, manifest)
    assert installer.checkpoint_contract(card) == configuration["checkpoints"]["qad-step5500-ple1000"]["model"]
    assert installer.checkpoint_contract(derived) == configuration["checkpoints"][NAME]["model"]
    other = derived_checkpoint.donor_card(card, manifest)
    assert (other["target_variant"], other["model_revision"]) == ("qad-step-4000", DONOR)
    assert installer.checkpoint_pins(other) == donor
    assert derived_checkpoint.model_of(card) == configuration["checkpoints"][NAME]["model"]
    assert derived_checkpoint.model_of(setup.selection(PROFILES[0])) is None
    assert derived_checkpoint.load(setup.selection(PROFILES[0], "qad-step-4000")) is None


def test_a_derived_checkpoint_counts_its_recipe_files_and_donor_files_in_storage_planning():
    from runtime.host import checkpoint_plan
    card, _, manifest, base, donor = inputs()
    base_sizes = [entry["size"] for name, entry in base["files"].items() if name not in base["optional"]]
    expected = checkpoint_plan.required_space(base_sizes + list(derived_checkpoint.files_of(manifest, "recipe").values())
                                              + [donor["files"][name]["size"] for name in manifest["donor"]["files"]])
    assert setup.pinned_checkpoint_bytes(card) == expected
    assert setup.pinned_checkpoint_bytes(setup.selection(PROFILES[0])) == checkpoint_plan.required_space(base_sizes)
    assert expected - checkpoint_plan.required_space(base_sizes) == 6_021_571_547 + 2_786_893_032 + 33_295_470


def lock(profile, checkpoint=NAME):
    nodes = 4 if profile.endswith("-tp4") else 2
    raw = install_site(nodes)
    owned = f"/srv/sparkring/parity/checkpoints/{REPOSITORY.replace('/', '--')}/{BASE}"
    for row in raw["hosts"]:
        row["model"] = owned
    return installer.make_lock(profile, raw, "1" * 40, "2" * 64, checkpoint,
                               image_runtime=installer.installer_image.for_profile(profile))


def test_rendered_containers_mount_the_derived_directory_and_serve_its_name():
    for profile, suffix in zip(PROFILES, ("TP2", "TP4")):
        derived, default = lock(profile), lock(profile, None)
        assert derived["id"] != default["id"]
        manifest = derived_checkpoint.load(derived["selection"])
        directory = (f"/srv/sparkring/parity/checkpoints/sparkring-derived--Qwen3.8-Flash-Next-NVFP4-QAD5500-MXFP8-"
                     f"Attention/{manifest['revision']}")
        # The lock's rows keep the base's directory, which the checkpoint plan acquires.
        assert {row["model"] for row in derived["site"]["ranks"]} == {
            f"/srv/sparkring/parity/checkpoints/{REPOSITORY.replace('/', '--')}/{BASE}"}
        for spec, plain in zip(installer.specifications(derived), installer.specifications(default), strict=True):
            mounts = {mount.target: mount for mount in spec.mounts}
            assert (mounts["/models/target"].source, mounts["/models/target"].read_only) == (directory, True)
            served = spec.command[spec.command.index("--served-model-name") + 1]
            assert served == f"Qwen3.8-Flash-Next-NVFP4-QAD-MXFP8-Attention-{suffix}"
            # Only the served name, the model mount and the checkpoint-keyed cache paths differ from the default.
            assert [a for a in spec.command if a != served] == [a for a in plain.command
                                                                if a != f"Qwen3.8-Flash-Next-NVFP4-QAD-{suffix}"]
            changed = {key for key in set(spec.environment) | set(plain.environment)
                       if spec.environment.get(key) != plain.environment.get(key)}
            assert all(manifest["revision"][:12] in spec.environment[key] for key in changed) and changed
        assert installer.connection(derived)["model"] == f"Qwen3.8-Flash-Next-NVFP4-QAD-MXFP8-Attention-{suffix}"
        assert installer.identity(derived)["derived"] == {"repository": manifest["repository"],
                                                          "revision": manifest["revision"]}
        assert "derived" not in installer.identity(default)
        assert installer.served_model(derived, derived["site"]["ranks"][0]) == directory


def test_a_derived_checkpoint_needs_its_base_in_a_sparkring_checkpoint_directory():
    raw = install_site(2)
    for row in raw["hosts"]:
        row.update(model="/home/operator/qwen", reuse_verified_model=True)
    value = installer.make_lock(PROFILES[0], raw, "1" * 40, "2" * 64, NAME,
                                image_runtime=installer.installer_image.for_profile(PROFILES[0]))
    with pytest.raises(ValueError, match="never beside a named copy"):
        installer.specifications(value)


def test_the_checkpoint_table_refuses_malformed_derived_entries():
    profile = toolchain_profiles.read(installer.ROOT / "profiles/qwen38-flash-next-tp2/config.json")
    assert NAME in toolchain_profiles.checkpoint_names(profile)[1]
    settings = toolchain_profiles.checkpoint_settings(profile, NAME)
    assert settings["model"] == profile["checkpoints"][NAME]["model"]
    # The derived checkpoint keeps the profile's settings; only the served name differs.
    assert settings["environment"] == profile["environment"]
    assert settings["vllm_args"] == profile["vllm_args"]
    for change in ({"base": "qad-step5500-ple1000"}, {"base": "qad-step5500-ple1000", "donor": "qad-step5500-ple1000"},
                   {"base": "missing", "donor": "qad-step-4000"}, {"base": NAME, "donor": "qad-step-4000"}):
        changed = copy.deepcopy(profile)
        changed["checkpoints"][NAME]["derived"] = change
        with pytest.raises(ValueError, match="Derived checkpoint"):
            toolchain_profiles.checkpoint_names(changed)
    changed = copy.deepcopy(profile)
    changed["checkpoints"][NAME]["model"]["repository"] = REPOSITORY
    with pytest.raises(ValueError, match="Derived checkpoint"):
        toolchain_profiles.checkpoint_names(changed)
    changed = copy.deepcopy(profile)
    del changed["checkpoints"][NAME]["derived"]
    with pytest.raises(ValueError, match="without a derived object"):
        toolchain_profiles.checkpoint_names(changed)


def test_the_manifest_file_is_the_one_the_entry_names():
    card, configuration, manifest, _, _ = inputs()
    path = derived_checkpoint.manifest_path(configuration["checkpoints"][NAME]["model"])
    assert json.loads(path.read_text(encoding="utf-8")) == manifest
    assert path.relative_to(installer.ROOT).as_posix() == (
        f"profiles/checkpoints/sparkring-derived--Qwen3.8-Flash-Next-NVFP4-QAD5500-MXFP8-Attention/"
        f"{manifest['revision']}.json")


def test_the_derive_phase_follows_the_checkpoint_and_image_phases():
    derived, default = lock(PROFILES[0]), lock(PROFILES[0], None)
    for action in ("prepare", "up"):
        phases = [phase["id"] for phase in installer.operation_plan(derived, action)["phases"]]
        assert phases[:5] == ["prerequisites" if action == "up" else "prepare-prerequisites", "source", "model", "image",
                              "derive"]
        assert "derive" not in [phase["id"] for phase in installer.operation_plan(default, action)["phases"]]
    phase = next(item for item in installer.operation_plan(derived, "prepare")["phases"] if item["id"] == "derive")
    assert [action["verify"]["argv"][1] for action in phase["actions"]] == ["derive-check", "derive-check"]


def test_a_shared_export_of_a_derived_deployment_renders_its_example_containers(tmp_path):
    import zipfile
    value = lock(PROFILES[0])
    data = b"offline source bundle fixture"
    value = installer.make_lock(PROFILES[0], value["site_input"], "1" * 40, hashlib.sha256(data).hexdigest(), NAME,
                                image_runtime=value["image_runtime"])
    directory = tmp_path / "deployment"
    installer.write(directory / "deployment.lock.json", value)
    (directory / "source.bundle").write_bytes(data)
    result = installer.export(directory, tmp_path / "share.zip", share=True)
    assert "rank0/compose.yaml" in result["files"]
    with zipfile.ZipFile(tmp_path / "share.zip") as archive:
        site = json.loads(archive.read("site.example.json"))
        compose = archive.read("rank0/compose.yaml").decode()
    base = f"/srv/sparkring/example/checkpoints/{REPOSITORY.replace('/', '--')}/{BASE}"
    assert {row["model"] for row in site["hosts"]} == {base}
    manifest = derived_checkpoint.load(value["selection"])
    assert derived_checkpoint.directory(base, manifest) in compose
