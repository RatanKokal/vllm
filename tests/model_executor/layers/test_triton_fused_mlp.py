# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.triton_fused_mlp import (
    can_use_fused_gate_up_silu_mul,
    fused_gate_up_silu_mul,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.skipif(not HAS_TRITON, reason="Triton is required")
@pytest.mark.skipif(
    not current_platform.is_device_capability(75),
    reason="sm75-only fused path",
)
def test_fused_gate_up_silu_mul_matches_reference():
    torch.manual_seed(0)

    m, k, n = 96, 128, 192
    x = torch.randn((m, k), device="cuda", dtype=torch.float16)
    w = torch.randn((2 * n, k), device="cuda", dtype=torch.float16)

    assert can_use_fused_gate_up_silu_mul(x, w)

    out = fused_gate_up_silu_mul(x, w)

    gate = x @ w[:n, :].transpose(0, 1)
    up = x @ w[n:, :].transpose(0, 1)
    ref = F.silu(gate) * up

    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=2e-2, rtol=2e-2)
