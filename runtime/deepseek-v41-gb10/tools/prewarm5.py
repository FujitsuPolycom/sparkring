"""Prepare sparse MLA under runtime flags and require the MXFP8 GEMM cache."""
import os
import sys
import time


def main():
    # These flags change compiler cache keys; serving uses their unset values.
    for name in ("FLASHINFER_JIT_VERBOSE", "FLASHINFER_JIT_DEBUG", "FLASHINFER_JIT_LINEINFO"):
        os.environ.pop(name, None)
    started = time.monotonic()
    try:
        from flashinfer.mla._sparse_mla_sm120 import get_sparse_mla_sm120_module
        get_sparse_mla_sm120_module()
        print(f"SPARSE-BUILT+LOADED {time.monotonic() - started:.1f}s", flush=True)
        from flashinfer.jit.gemm import gen_gemm_sm120_module_cutlass_mxfp8
        compiled = gen_gemm_sm120_module_cutlass_mxfp8().is_compiled
        compiled = compiled() if callable(compiled) else compiled
        if not compiled:
            raise RuntimeError("MXFP8 GEMM cache is absent under the runtime environment")
        print("MXFP8-COMPILED", flush=True)
    except Exception as error:
        # A load error alone cannot establish that compilation succeeded.
        print(f"FlashInfer prewarm failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
