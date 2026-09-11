import os, time
for k in ("FLASHINFER_JIT_VERBOSE", "FLASHINFER_JIT_DEBUG", "FLASHINFER_JIT_LINEINFO"):
    os.environ.pop(k, None)   # these switch nvcc to debug/lineinfo flags -> cache mismatch vs the launcher
t = time.time()
from flashinfer.mla._sparse_mla_sm120 import get_sparse_mla_sm120_module as g
try:
    g(); print("SPARSE-BUILT+LOADED %.1fs" % (time.time() - t), flush=True)
except Exception as e:
    print("sparse build done, load raised (expected without GPU):", type(e).__name__, str(e)[:160], flush=True)
from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_mxfp8 as g1
ic = getattr(g1(), "is_compiled"); ic = ic() if callable(ic) else ic
print("mxfp8 is_compiled under runtime env:", ic, flush=True)
