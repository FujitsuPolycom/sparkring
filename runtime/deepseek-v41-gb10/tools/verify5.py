import time
t = time.time()
from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_mxfp8 as g1
r = g1().try_load()
print("VERIFY mxfp8: %s %.1fs" % ("HIT" if r is not None else "MISS", time.time() - t), flush=True)
t = time.time()
from flashinfer.mla._sparse_mla_sm120 import get_sparse_mla_sm120_module as g2
g2(); dt = time.time() - t
print("VERIFY sparse_mla: %s %.1fs" % ("HIT" if dt < 15 else "MISS-COMPILED", dt), flush=True)
