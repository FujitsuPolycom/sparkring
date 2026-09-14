"""BF16 Qwen HC up-projection with fused four-stream gating and mean."""

import triton
import triton.language as tl


@triton.jit
def kernel(
    X,
    W,
    N,
    Y,
    M,
    H: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.arange(0, 4 * BH)
    hidden = tl.program_id(1) * BH + cols // 4
    expanded = (cols % 4) * H + hidden
    ks = tl.arange(0, BK)
    acc = tl.zeros((BM, 4 * BH), tl.float32)
    for block in range(tl.cdiv(K, BK)):
        k = block * BK + ks
        a = tl.load(
            X + rows[:, None] * K + k[None, :],
            (rows[:, None] < M) & (k[None, :] < K),
            other=0,
        )
        b = tl.load(
            W + expanded[None, :] * K + k[:, None],
            (hidden[None, :] < H) & (k[:, None] < K),
            other=0,
        )
        acc = tl.dot(a, b, acc)
    logits = acc.to(tl.bfloat16).to(tl.float32)
    gate = tl.sigmoid(logits).to(tl.bfloat16)
    normalized = tl.load(
        N + rows[:, None] * (4 * H) + expanded[None, :],
        (rows[:, None] < M) & (hidden[None, :] < H),
        other=0,
    )
    product = (gate * normalized).to(tl.bfloat16).to(tl.float32)
    even, odd = tl.split(tl.reshape(product, (BM, BH, 2, 2)))
    p0, p2 = tl.split(even)
    p1, p3 = tl.split(odd)
    total = ((p0 + p1) + p2) + p3
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    tl.store(
        Y + rows[:, None] * H + h[None, :],
        total / 4,
        (rows[:, None] < M) & (h[None, :] < H),
    )


def fused(x, w, n, y, bm, bh, bk, warps):
    kernel[(triton.cdiv(x.shape[0], bm), triton.cdiv(2560, bh))](
        x,
        w,
        n,
        y,
        x.shape[0],
        2560,
        320,
        bm,
        bh,
        bk,
        num_warps=warps,
        num_stages=3,
        enable_fp_fusion=False,
    )
    return y
