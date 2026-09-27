import ctypes
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import tempfile

subprocess.run([sys.executable, '/opt/sparkring/toolchain/toolchain.py', 'verify'], check=True)
import torch

assert torch.cuda.is_available()
device = torch.cuda.get_device_properties(0)
assert (device.major, device.minor) == (12, 1)
nccl = ctypes.CDLL('/opt/sparkring/toolchain/nccl/lib/libnccl.so.2')
nccl_version = ctypes.c_int()
assert nccl.ncclGetVersion(ctypes.byref(nccl_version)) == 0
assert nccl_version.value == 23203
modules = ['vllm._C_stable_libtorch', 'vllm._moe_C_stable_libtorch',
           'vllm._flashkda_C', 'vllm.vllm_flash_attn._vllm_fa2_C',
           'flashinfer', 'b12x', 'cutlass.cute', 'cuda.bindings.driver']
for name in modules:
    importlib.import_module(name)
value = torch.ones((32, 32), device='cuda', dtype=torch.float16)
product = value @ value
torch.cuda.synchronize()
assert bool(torch.all(product == 32))

driver = ctypes.CDLL('/usr/local/cuda-13.4/compat/libcuda.so.1')
version = ctypes.c_int()
assert driver.cuDriverGetVersion(ctypes.byref(version)) == 0
cudart = ctypes.CDLL('/usr/local/cuda-13.4/lib64/libcudart.so.13')
runtime_version = ctypes.c_int()
assert cudart.cudaRuntimeGetVersion(ctypes.byref(runtime_version)) == 0
assert runtime_version.value == 13040

with tempfile.TemporaryDirectory() as temp:
    source = Path(temp) / 'smoke.cu'
    ptx = Path(temp) / 'smoke.ptx'
    source.write_text('extern "C" __global__ void increment(int* p) { p[threadIdx.x] += 1; }\n')
    subprocess.run(['/usr/local/cuda/bin/nvcc', '--ptx', '-arch=compute_121',
                    str(source), '-o', str(ptx)], check=True)
    tensor = torch.arange(128, dtype=torch.int32, device='cuda')
    torch.cuda.synchronize()
    module = ctypes.c_void_p()
    driver.cuModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    assert driver.cuModuleLoad(ctypes.byref(module), str(ptx).encode()) == 0
    function = ctypes.c_void_p()
    driver.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
    assert driver.cuModuleGetFunction(ctypes.byref(function), module, b'increment') == 0
    pointer = ctypes.c_uint64(tensor.data_ptr())
    args = (ctypes.c_void_p * 1)(ctypes.cast(ctypes.byref(pointer), ctypes.c_void_p))
    driver.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    assert driver.cuLaunchKernel(function, 1, 1, 1, 128, 1, 1, 0, None, args, None) == 0
    assert driver.cuCtxSynchronize() == 0
    assert torch.equal(tensor.cpu(), torch.arange(1, 129, dtype=torch.int32))
    driver.cuModuleUnload.argtypes = [ctypes.c_void_p]
    assert driver.cuModuleUnload(module) == 0
    ptx_digest = hashlib.sha256(ptx.read_bytes()).hexdigest()
paths = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                if any(name in line for name in ('libcuda.', 'libcudart.', 'libcublas', 'libnccl.',
                    'libnvrtc', 'libnvJitLink', 'libcufft', 'libcusolver', 'libcusparse.', 'libcurand'))})
nccl_paths = [path for path in paths if 'libnccl.' in path]
assert len(nccl_paths) == 1 and nccl_paths[0].startswith('/opt/sparkring/toolchain/nccl/'), nccl_paths
assert all(path.startswith('/usr/local/cuda-13.4/') for path in paths if 'libnccl.' not in path), paths
print(json.dumps({'status':'bounded-gpu-smoke-passed', 'device':device.name,
    'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'compute_capability':[device.major,device.minor], 'torch':torch.__version__,
    'torch_build_cuda':torch.version.cuda, 'selected_cuda_runtime':runtime_version.value,
    'selected_cuda_driver_api':version.value, 'nccl_runtime_version':nccl_version.value,
    'torch_reported_nccl_version':list(torch.cuda.nccl.version()),
    'imported_modules':modules, 'ptx_sha256':ptx_digest, 'mapped_libraries':paths,
    'cuda_matmul_correct':True, 'explicit_ptx_jit_correct':True,
    'gpu_tensor_peak_bytes':torch.cuda.max_memory_allocated(),
    'multi_rank_tested':False, 'model_serving_tested':False},indent=2))
