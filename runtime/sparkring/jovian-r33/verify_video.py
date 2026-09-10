"""Decode one deterministic H.264 fixture through the R33 ARM media backends."""

import json
from pathlib import Path

import torch
from torchcodec.decoders import VideoDecoder
import PyNvVideoCodec as nvc


path = Path("/fixture/blue-h264.mp4")
data = path.read_bytes()

cpu_decoder = VideoDecoder(data, dimension_order="NHWC", device="cpu")
cpu_frames = cpu_decoder.get_all_frames().data
assert cpu_frames.ndim == 4 and cpu_frames.shape[-1] == 3
cpu_means = cpu_frames.to(torch.float32).mean(dim=(0, 1, 2))
assert cpu_means[2] > 200 and cpu_means[0] < 30 and cpu_means[1] < 30, cpu_means

gpu_decoder = nvc.SimpleDecoder(
    str(path),
    gpu_id=0,
    use_device_memory=False,
    output_color_type=nvc.OutputColorType.RGB,
    bWaitForSessionWarmUp=True,
)
assert len(gpu_decoder) >= 1
gpu_frame = torch.from_dlpack(gpu_decoder[0])
assert gpu_frame.ndim == 3 and gpu_frame.shape[-1] == 3, gpu_frame.shape
gpu_means = gpu_frame.to(torch.float32).mean(dim=(0, 1))
assert gpu_means[2] > 200 and gpu_means[0] < 30 and gpu_means[1] < 30, gpu_means
print(
    json.dumps(
        {
            "status": "qualified-cpu-and-gpu-video-decode",
            "fixture": path.name,
            "fixture_bytes": len(data),
            "torchcodec_frames": len(cpu_frames),
            "torchcodec_rgb_mean": cpu_means.tolist(),
            "pynvvideocodec_frames": len(gpu_decoder),
            "pynvvideocodec_rgb_mean": gpu_means.tolist(),
            "limits": "One 224x224 H.264 fixture; encode and additional codecs remain untested.",
        },
        indent=2,
    )
)
