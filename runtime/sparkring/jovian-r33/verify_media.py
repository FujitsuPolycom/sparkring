"""Check R33 ARM media imports and a CPU TorchAudio operation."""

import importlib.metadata
import json

import importlib

import torch


assert torch.__version__ == "2.13.0"
versions = {
    "torchvision": "0.28.0",
    "torchaudio": "2.11.0+cu130",
    "torchcodec": "0.16.0+cu130",
    "PyNvVideoCodec": "2.0.4",
}
imports = {}
modules = {}
for name, expected in versions.items():
    try:
        modules[name] = importlib.import_module(name)
        actual = importlib.metadata.version(name)
        assert actual == expected, (name, actual, expected)
        imports[name] = "passed"
    except Exception as error:  # report every independent native compatibility result
        imports[name] = f"{type(error).__name__}: {error}"
if "torchaudio" in modules:
    wave = torch.linspace(-1.0, 1.0, 1600).reshape(1, -1)
    resampled = modules["torchaudio"].functional.resample(wave, 16000, 8000)
    assert resampled.shape == (1, 800)
    assert torch.isfinite(resampled).all()
print(
    json.dumps(
        {
            "status": "qualified" if all(value == "passed" for value in imports.values()) else "failed",
            "imports": imports,
            "cuda_available": torch.cuda.is_available(),
            "limits": "Imports and CPU resampling only; video/JPEG decode and encode remain pending.",
        },
        indent=2,
    )
)
assert all(value == "passed" for value in imports.values()), imports
