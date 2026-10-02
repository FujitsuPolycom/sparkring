"""The MXFP8-attention recipe on small synthetic checkpoints.

A base with BF16 attention projections and a donor whose MXFP8 tensors are the
base's own weights quantized by ``reference_quantize`` (a stand-in with the
interface of vLLM's ``_mxfp8_e4m3_quantize_torch``) are written as safetensors
files. The recipe must reproduce the donor bytes before it writes anything for a
shard, and its output must not depend on the run.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from runtime.common import derived_checkpoint
from runtime.common import mxfp8_attention as recipe

torch = pytest.importorskip("torch")

GDN = "model.language_model.layers.0.linear_attn."
FULL = "model.language_model.layers.3.self_attn."
MODULES = sorted([GDN + "in_proj_a", GDN + "in_proj_qkv", GDN + "out_proj", FULL + "q_proj",
                  FULL + "indexer.index_qk_proj"])
CHANGED = "model-00001-of-00002.safetensors"
KEPT = "model-00002-of-00002.safetensors"
DONOR = "model-00009-of-00009.safetensors"


def reference_quantize(tensor, swizzled=False):
    """MXFP8 E4M3 with one power-of-two UE8M0 scale per 32 values along the last dimension."""
    assert not swizzled
    blocks = tensor.float().reshape(*tensor.shape[:-1], tensor.shape[-1] // 32, 32)
    exponent = torch.floor(torch.log2(blocks.abs().amax(-1, keepdim=True).clamp(min=2.0 ** -120))) - 8
    weight = (blocks / torch.pow(2.0, exponent)).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(tensor.shape)
    return weight, (exponent + 127).to(torch.uint8).reshape(*tensor.shape[:-1], tensor.shape[-1] // 32)


def raw(tensor):
    return tensor.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes() if tensor.numel() else b""


def quantize(data, shape):
    weight, scale = reference_quantize(torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(shape))
    return list(weight.shape), raw(weight), raw(scale)


def shard(path, tensors, metadata=None):
    names = {torch.bfloat16: "BF16", torch.int64: "I64", torch.float8_e4m3fn: "F8_E4M3", torch.uint8: "U8",
             torch.float32: "F32"}
    recipe.write_shard(path, {name: (names[t.dtype], list(t.shape), raw(t)) for name, t in tensors.items()},
                       metadata if metadata is not None else {"format": "pt"})


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=4) + "\n", encoding="utf-8")


def fixture(root):
    """``(base, donor, record)`` directories and the record the recipe must find."""
    generator = torch.Generator().manual_seed(7)
    base, donor = Path(root) / "base", Path(root) / "donor"
    base.mkdir(parents=True)
    donor.mkdir()
    weights = {module + ".weight": torch.randn(16 * (1 + i % 3), 64, generator=generator).to(torch.bfloat16)
               for i, module in enumerate(MODULES)}
    changed = {**weights, GDN + "A_log": torch.randn(8, generator=generator).to(torch.bfloat16),
               "model.language_model.layers.0.ple.offsets": torch.arange(4, dtype=torch.int64),
               FULL + "q_norm.weight": torch.ones(64, dtype=torch.bfloat16)}
    kept = {"model.language_model.layers.1.mlp.weight": torch.randn(32, 32, generator=generator).to(torch.bfloat16),
            "model.language_model.layers.1.mlp.weight_scale_2": torch.tensor(0.5)}
    shard(base / CHANGED, changed)
    shard(base / KEPT, kept)
    weight_map = {name: CHANGED for name in changed} | {name: KEPT for name in kept}
    write_json(base / recipe.INDEX, {"metadata": {"total_size": 1}, "weight_map": weight_map})
    write_json(base / "config.json", {"architectures": ["Fixture"], "quantization_config": {
        "quant_method": "modelopt", "quantized_layers": {"model.language_model.layers.1.mlp": {"quant_algo": "NVFP4"}}}})
    write_json(base / "hf_quant_config.json", {"quant_method": "modelopt", "quantized_layers": {}})
    quantized = {}
    for name, tensor in weights.items():
        weight, scale = reference_quantize(tensor)
        quantized[name] = weight
        quantized[name[:-len(".weight")] + ".weight_scale"] = scale
    shard(donor / DONOR, quantized)
    write_json(donor / recipe.INDEX, {"metadata": {}, "weight_map": {name: DONOR for name in quantized}})
    record = {"schema": derived_checkpoint.RECORD_SCHEMA, "checkpoint": {"repository": "sparkring-derived/fixture",
                                                                         "revision": "f" * 40},
              "format": dict(recipe.ENTRY), "modules": {"count": len(MODULES),
                                                         "sha256": recipe.modules_digest(MODULES)},
              "rewritten_shards": [CHANGED]}
    return base, donor, record


def tree(directory):
    return {path.name: path.read_bytes() for path in sorted(Path(directory).iterdir())}


def derive(tmp_path, name="out", **changes):
    base, donor, record = fixture(tmp_path / ("in-" + name))
    out = tmp_path / name
    out.mkdir()
    sizes = recipe.derive(str(base), str(donor), str(out), {**record, **changes}, quantize, log=lambda text: None)
    return base, donor, out, record, sizes


def test_recipe_replaces_each_projection_after_the_byte_exact_check(tmp_path):
    base, donor, out, record, sizes = derive(tmp_path)
    assert sorted(os.listdir(out)) == sorted([CHANGED, "config.json", "hf_quant_config.json", recipe.INDEX,
                                              recipe.RECORD])
    metadata, tensors, start = recipe.read_header(out / CHANGED)
    _, donor_tensors, donor_start = recipe.read_header(donor / DONOR)
    _, base_tensors, base_start = recipe.read_header(base / CHANGED)
    assert metadata == {"format": "pt"}
    for module in MODULES:
        for suffix, dtype in ((".weight", "F8_E4M3"), (".weight_scale", "U8")):
            name = module + suffix
            assert tensors[name][0] == dtype and tensors[name][1] == donor_tensors[name][1]
            assert (recipe.read_tensor(out / CHANGED, start, *tensors[name][2:])
                    == recipe.read_tensor(donor / DONOR, donor_start, *donor_tensors[name][2:]))
    # Every other tensor keeps its dtype, shape and bytes.
    for name, (dtype, shape, begin, end) in base_tensors.items():
        if name[:-len(".weight")] not in MODULES:
            assert tensors[name][:2] == (dtype, shape)
            assert (recipe.read_tensor(out / CHANGED, start, *tensors[name][2:])
                    == recipe.read_tensor(base / CHANGED, base_start, begin, end))
    config = json.loads((out / "config.json").read_text())
    assert config["quantization_config"]["quantized_layers"] == {
        "model.language_model.layers.1.mlp": {"quant_algo": "NVFP4"}, **{m: recipe.ENTRY for m in MODULES}}
    assert json.loads((out / "hf_quant_config.json").read_text())["quantized_layers"] == {m: recipe.ENTRY
                                                                                         for m in MODULES}
    index = json.loads((out / recipe.INDEX).read_text())
    assert index["metadata"]["total_size"] == (out / CHANGED).stat().st_size + (base / KEPT).stat().st_size
    assert all(index["weight_map"][m + ".weight_scale"] == CHANGED for m in MODULES)
    assert list(index["weight_map"]) == sorted(index["weight_map"])
    assert (out / recipe.RECORD).read_bytes() == derived_checkpoint.record_bytes(record)
    assert sizes == {name: (out / name).stat().st_size for name in os.listdir(out)}


def test_recipe_output_does_not_depend_on_the_run(tmp_path):
    *_, first, _, _ = derive(tmp_path, "first")
    *_, second, _, _ = derive(tmp_path, "second")
    assert tree(first) == tree(second)


def test_recipe_refuses_a_donor_that_is_not_the_base_in_mxfp8(tmp_path):
    base, donor, record = fixture(tmp_path / "in")
    metadata, tensors, start = recipe.read_header(donor / DONOR)
    changed = MODULES[2] + ".weight_scale"
    sources = {name: (dtype, shape, recipe.read_tensor(donor / DONOR, start, begin, end))
               for name, (dtype, shape, begin, end) in tensors.items()}
    scale = bytearray(sources[changed][2])
    scale[0] ^= 1
    sources[changed] = (*sources[changed][:2], bytes(scale))
    (donor / DONOR).unlink()
    recipe.write_shard(donor / DONOR, sources, metadata)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(recipe.RecipeError, match=MODULES[2].replace(".", r"\.") + ": quantizing the base"):
        recipe.derive(str(base), str(donor), str(out), record, quantize, log=lambda text: None)
    assert os.listdir(out) == []


@pytest.mark.parametrize("change, message", [
    ({"modules": {"count": 4, "sha256": "0" * 64}}, "does not name"),
    ({"rewritten_shards": [KEPT]}, "not in the record's rewritten shards"),
    ({"format": {"quant_algo": "MXFP8", "group_size": 16}}, "another format"),
])
def test_recipe_refuses_a_record_that_names_other_projections(tmp_path, change, message):
    with pytest.raises(recipe.RecipeError, match=message):
        derive(tmp_path, **change)


def test_recipe_refuses_inputs_without_the_pinned_layout(tmp_path):
    base, donor, record = fixture(tmp_path / "in")
    index = json.loads((donor / recipe.INDEX).read_text())
    del index["weight_map"][MODULES[0] + ".weight_scale"]
    write_json(donor / recipe.INDEX, index)
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(recipe.RecipeError, match="donor lacks " + MODULES[0].replace(".", r"\.")):
        recipe.derive(str(base), str(donor), str(out), record, quantize)
    (out / "stale").write_bytes(b"")
    with pytest.raises(recipe.RecipeError, match="is not empty"):
        recipe.derive(str(base), str(donor), str(out), record, quantize)


def test_writer_matches_the_safetensors_layout():
    """The header form of safetensors.torch.save_file 0.7.0 for one metadata entry, recorded as bytes."""
    tensors = {"b.weight": ("BF16", [2, 2], bytes(range(8))), "a.idx": ("I64", [1], bytes(8)),
               "c.weight": ("F8_E4M3", [2], b"\x01\x02"), "c.weight_scale": ("U8", [1], b"\x7f"),
               "d.scalar": ("F32", [], bytes(4))}
    header, order = recipe.layout({name: (dtype, shape, len(data)) for name, (dtype, shape, data) in tensors.items()},
                                  {"format": "pt"})
    assert order == ["a.idx", "d.scalar", "b.weight", "c.weight", "c.weight_scale"]
    assert header == (b"\x60\x01\x00\x00\x00\x00\x00\x00"
                      b'{"__metadata__":{"format":"pt"},"a.idx":{"dtype":"I64","shape":[1],"data_offsets":[0,8]},'
                      b'"d.scalar":{"dtype":"F32","shape":[],"data_offsets":[8,12]},'
                      b'"b.weight":{"dtype":"BF16","shape":[2,2],"data_offsets":[12,20]},'
                      b'"c.weight":{"dtype":"F8_E4M3","shape":[2],"data_offsets":[20,22]},'
                      b'"c.weight_scale":{"dtype":"U8","shape":[1],"data_offsets":[22,23]}}     ')
    # Several metadata entries keep a fixed order, which safetensors' own writer does not.
    assert recipe.layout({}, {"z": "1", "a": "2"})[0][8:] == b'{"__metadata__":{"a":"2","z":"1"}}      '


def test_writer_equals_safetensors_when_the_library_is_installed(tmp_path):
    safetensors = pytest.importorskip("safetensors.torch")
    generator = torch.Generator().manual_seed(1)
    tensors = {"w": torch.randn(8, 32, generator=generator).to(torch.bfloat16), "i": torch.arange(3),
               "q": torch.randn(4, 32, generator=generator).to(torch.float8_e4m3fn),
               "s": torch.randint(0, 255, (4, 1), dtype=torch.uint8, generator=generator), "x": torch.tensor(1.5)}
    safetensors.save_file(tensors, tmp_path / "library.safetensors", metadata={"format": "pt"})
    shard(tmp_path / "recipe.safetensors", tensors)
    assert (tmp_path / "library.safetensors").read_bytes() == (tmp_path / "recipe.safetensors").read_bytes()


STUB = '''
import torch


def _mxfp8_e4m3_quantize_torch(tensor, swizzled):
    blocks = tensor.float().reshape(*tensor.shape[:-1], tensor.shape[-1] // 32, 32)
    exponent = torch.floor(torch.log2(blocks.abs().amax(-1, keepdim=True).clamp(min=2.0 ** -120))) - 8
    weight = (blocks / torch.pow(2.0, exponent)).clamp(-448, 448).to(torch.float8_e4m3fn).reshape(tensor.shape)
    return weight, (exponent + 127).to(torch.uint8).reshape(*tensor.shape[:-1], tensor.shape[-1] // 32)
'''


def test_recipe_runs_as_program_text_with_the_vllm_quantizer(tmp_path):
    """The installer runs the file's text with python3 -c; a stub vLLM module stands in for the image's."""
    base, donor, record = fixture(tmp_path / "in")
    package = tmp_path / "stub" / "vllm/model_executor/layers/quantization/utils"
    package.mkdir(parents=True)
    for parent in [package, *package.parents][:5]:
        (parent / "__init__.py").write_text("")
    (package / "mxfp8_utils.py").write_text(STUB)
    out = tmp_path / "out"
    out.mkdir()
    environment = {**os.environ, "PYTHONPATH": os.pathsep.join([str(tmp_path / "stub"), *sys.path])}
    source = Path(recipe.__file__).read_text(encoding="utf-8")
    done = subprocess.run([sys.executable, "-c", source, "--base", str(base), "--donor", str(donor), "--out", str(out),
                           "--record", json.dumps(record)], capture_output=True, text=True, env=environment)
    assert done.returncode == 0, done.stderr
    assert "rewrote " + CHANGED in done.stdout
    expected = tmp_path / "expected"
    expected.mkdir()
    recipe.derive(str(base), str(donor), str(expected), record, quantize, log=lambda text: None)
    assert tree(out) == tree(expected)
    record["modules"] = {"count": 1, "sha256": "0" * 64}
    failed = subprocess.run([sys.executable, "-c", source, "--base", str(base), "--donor", str(donor),
                             "--out", str(tmp_path / "stub"), "--record", json.dumps(record)],
                            capture_output=True, text=True, env=environment)
    assert failed.returncode == 1 and failed.stderr.startswith("mxfp8_attention: ")


def test_module_digest_of_the_qwen_projections():
    """The 240 projection names of the Qwen checkpoints hash to the digest the derived manifest pins."""
    modules = []
    for layer in range(48):
        if layer % 4 == 3:
            modules += [f"model.language_model.layers.{layer}.self_attn.{p}"
                        for p in ("q_proj", "k_proj", "v_proj", "o_proj", "indexer.index_qk_proj")]
        else:
            modules += [f"model.language_model.layers.{layer}.linear_attn.{p}"
                        for p in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj")]
    names = sorted(name + ".weight" for name in modules)
    assert all(recipe.TARGET.match(name) for name in names)
    ordered = [name[:-len(".weight")] for name in names]
    assert hashlib.sha256("\n".join(ordered).encode()).hexdigest() == (
        "24bac2b336bc7755dd5e055f09289797284ad9a2b548122886d8bc73051a0504")
