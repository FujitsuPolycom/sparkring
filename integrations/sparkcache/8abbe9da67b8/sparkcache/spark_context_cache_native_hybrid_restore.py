"""Compatibility aliases for the SparkCache CUDA hybrid-restore interface."""

from sparkcache.spark_context_cache_cuda_hybrid_restore import (
    CudaAuthenticatedPageObject,
    CudaHybridRestoreError,
    CudaHybridRestoreResult,
    CudaPageDeltaRestorePlan,
    CudaPageObject,
    CudaPageRestoreError,
    CudaPageRestoreResult,
    CudaPageSlab,
    CudaPageSourceSpan,
    build_page_copy_spans,
    build_page_object_spans,
    execute_cuda_direct_restore,
    execute_cuda_hybrid_placement,
    execute_cuda_hybrid_restore,
    execute_cuda_page_placement,
    plan_cuda_page_delta_restore,
    plan_page_slabs,
)

NativeHybridRestoreError = CudaHybridRestoreError
NativeHybridRestoreResult = CudaHybridRestoreResult
NativePageDeltaRestorePlan = CudaPageDeltaRestorePlan
NativePageObject = CudaPageObject
NativePageSlab = CudaPageSlab
NativePageSourceSpan = CudaPageSourceSpan
execute_native_hybrid_placement = execute_cuda_hybrid_placement
execute_native_hybrid_restore = execute_cuda_hybrid_restore
plan_native_page_delta_restore = plan_cuda_page_delta_restore


__all__ = [
    "CudaAuthenticatedPageObject",
    "CudaHybridRestoreError",
    "CudaHybridRestoreResult",
    "CudaPageDeltaRestorePlan",
    "CudaPageObject",
    "CudaPageRestoreError",
    "CudaPageRestoreResult",
    "CudaPageSlab",
    "CudaPageSourceSpan",
    "NativeHybridRestoreError",
    "NativeHybridRestoreResult",
    "NativePageDeltaRestorePlan",
    "NativePageObject",
    "NativePageSlab",
    "NativePageSourceSpan",
    "build_page_copy_spans",
    "build_page_object_spans",
    "execute_cuda_direct_restore",
    "execute_cuda_hybrid_placement",
    "execute_cuda_hybrid_restore",
    "execute_cuda_page_placement",
    "execute_native_hybrid_placement",
    "execute_native_hybrid_restore",
    "plan_cuda_page_delta_restore",
    "plan_native_page_delta_restore",
    "plan_page_slabs",
]
