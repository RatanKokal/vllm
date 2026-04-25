# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental Triton kernels for fused MLP paths."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton


_SM75_FUSED_GATE_UP_CONFIGS = [
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 32, "BLOCK_N": 128, "BLOCK_K": 32},
        num_warps=4,
        num_stages=2,
    ),
    triton.Config(
        {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64},
        num_warps=8,
        num_stages=2,
    ),
]


@triton.autotune(configs=_SM75_FUSED_GATE_UP_CONFIGS, key=["M", "N", "K"])
@triton.jit
def _fused_gate_up_silu_mul_kernel(
    x_ptr,
    w_ptr,
    o_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wm,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc_gate = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_up = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)

        # Weight layout is [2N, K]. We compute:
        # gate = x @ w[:N, :].T and up = x @ w[N:, :].T in a single pass.
        w_gate_ptrs = (
            w_ptr
            + offs_n[None, :] * stride_wm
            + offs_k[:, None] * stride_wk
        )
        w_up_ptrs = (
            w_ptr
            + (offs_n + N)[None, :] * stride_wm
            + offs_k[:, None] * stride_wk
        )
        w_gate = tl.load(
            w_gate_ptrs,
            mask=k_mask[:, None] & (offs_n[None, :] < N),
            other=0.0,
        )
        w_up = tl.load(
            w_up_ptrs,
            mask=k_mask[:, None] & (offs_n[None, :] < N),
            other=0.0,
        )

        acc_gate = tl.dot(x, w_gate, acc=acc_gate)
        acc_up = tl.dot(x, w_up, acc=acc_up)

        offs_k += BLOCK_K

    gate = acc_gate * tl.sigmoid(acc_gate)
    out = gate * acc_up

    o_ptrs = o_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(
        o_ptrs,
        out.to(o_ptr.type.element_ty),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def can_use_fused_gate_up_silu_mul(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
) -> bool:
    if not HAS_TRITON:
        return False
    if bias is not None:
        return False
    if not current_platform.is_cuda_alike():
        return False
    if not current_platform.is_device_capability(75):
        return False
    if not x.is_cuda or not weight.is_cuda:
        return False
    if x.ndim < 2:
        return False
    if x.shape[-1] != weight.shape[-1]:
        return False
    if weight.ndim != 2:
        return False
    if weight.shape[0] % 2 != 0:
        return False
    if x.dtype not in (torch.float16, torch.bfloat16):
        return False
    if weight.dtype not in (torch.float16, torch.bfloat16):
        return False
    return True


def fused_gate_up_silu_mul(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute silu(x @ W_gate^T + b_gate) * (x @ W_up^T + b_up).

    Expects ``weight`` to be shaped [2N, K] in [gate, up] layout.
    """
    if not can_use_fused_gate_up_silu_mul(x, weight, bias=bias):
        raise ValueError("fused_gate_up_silu_mul received unsupported inputs")

    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    w = weight.contiguous()

    m = x_2d.shape[0]
    n = w.shape[0] // 2
    k = x_2d.shape[1]

    out = torch.empty((m, n), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(m, meta["BLOCK_M"]), triton.cdiv(n, meta["BLOCK_N"]))
    _fused_gate_up_silu_mul_kernel[grid](
        x_2d,
        w,
        out,
        m,
        n,
        k,
        x_2d.stride(0),
        x_2d.stride(1),
        w.stride(0),
        w.stride(1),
        out.stride(0),
        out.stride(1),
    )

    return out.view(*x.shape[:-1], n)
