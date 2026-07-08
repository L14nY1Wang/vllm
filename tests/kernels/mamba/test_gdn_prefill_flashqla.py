# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

# FlashQLA targets Hopper (SM90), Blackwell (SM10.x) and Blackwell SM12.x;
# its arch dispatch raises ValueError elsewhere. Skip before importing it.
if not (
    current_platform.is_cuda()
    and (
        current_platform.is_device_capability(90)
        or current_platform.is_device_capability_family(100)
        or current_platform.is_device_capability_family(120)
    )
):
    pytest.skip(
        reason="GDN FlashQLA prefill requires CUDA SM90/SM10x/SM12x.",
        allow_module_level=True,
    )

try:
    import flash_qla  # noqa: F401
except Exception:
    pytest.skip(
        reason="flash_qla package not installed; skipping FlashQLA GDN tests.",
        allow_module_level=True,
    )

from tests.v1.attention.utils import create_vllm_config  # noqa: E402
from vllm.config import set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.fla.ops import (  # noqa: E402
    chunk_gated_delta_rule,
)
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (  # noqa: E402
    ChunkGatedDeltaRule,
)


@pytest.mark.parametrize("num_seqs", [1, 5, 257])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_gdn_chunk_flashqla_correctness(num_seqs: int, state_dtype: torch.dtype):
    torch.manual_seed(0)
    seq_lens = torch.randint(1, 130, (num_seqs,), dtype=torch.int32)
    cu_seqlens = torch.zeros(num_seqs + 1, device="cuda", dtype=torch.int32)
    cu_seqlens[1:] = seq_lens.to(device="cuda").cumsum(0)
    total_tokens = int(cu_seqlens[-1].item())

    num_k_heads = 4
    num_v_heads = 8
    head_k_dim = 128
    head_v_dim = 128
    dtype = torch.bfloat16

    q = torch.randn(
        1, total_tokens, num_k_heads, head_k_dim, device="cuda", dtype=dtype
    )
    k = torch.randn_like(q)
    v = torch.randn(
        1, total_tokens, num_v_heads, head_v_dim, device="cuda", dtype=dtype
    )
    q = F.normalize(q.float(), p=2, dim=-1).to(dtype)
    k = F.normalize(k.float(), p=2, dim=-1).to(dtype)
    a = torch.randn(1, total_tokens, num_v_heads, device="cuda", dtype=dtype)
    b = torch.randn(1, total_tokens, num_v_heads, device="cuda", dtype=dtype)
    # Match upstream FLA GatedDeltaNet synthetic initialization.
    A = torch.empty(num_v_heads, device="cuda", dtype=torch.float32).uniform_(0, 16)
    A_log = torch.log(A)
    dt = torch.exp(
        torch.rand(num_v_heads, device="cuda", dtype=torch.float32)
        * (math.log(0.1) - math.log(0.001))
        + math.log(0.001)
    )
    dt = torch.clamp(dt, min=1e-4)
    dt_bias = dt + torch.log(-torch.expm1(-dt))
    g = -A_log.exp().view(1, 1, num_v_heads) * F.softplus(
        a.float() + dt_bias.view(1, 1, num_v_heads)
    )
    beta = torch.sigmoid(b.float())
    # FLA reference state layout is [N, num_v_heads, V, K]; FlashQLA with
    # state_v_first=True consumes exactly this layout.
    initial_state = (
        torch.randn(
            num_seqs,
            num_v_heads,
            head_v_dim,
            head_k_dim,
            device="cuda",
            dtype=state_dtype,
        )
        * 0.05
    )

    ref_o, ref_state = chunk_gated_delta_rule(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )

    # --- Direct flash_qla API call (mirrors forward_flashqla internals) ---
    with torch.no_grad():
        actual_o, actual_state = flash_qla.chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g.to(torch.float32),
            beta=beta.to(torch.float32),
            initial_state=initial_state.to(torch.float32),
            output_final_state=True,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=cu_seqlens,
            head_first=False,
            state_v_first=True,
            auto_cp=True,
        )
    torch.accelerator.synchronize()

    o_error = (actual_o.float() - ref_o.float()).abs()
    state_error = (
        actual_state.float() - ref_state.to(actual_state.dtype).float()
    ).abs()
    assert o_error.max().item() < 2e-2, f"o max error {o_error.max().item()}"
    assert o_error.mean().item() < 6e-4, f"o mean error {o_error.mean().item()}"
    assert state_error.max().item() < 2e-2, (
        f"state max error {state_error.max().item()}"
    )
    assert state_error.mean().item() < 6e-3, (
        f"state mean error {state_error.mean().item()}"
    )


def test_gdn_chunk_flashqla_core_attn_out(monkeypatch):
    """Exercise the ChunkGatedDeltaRule custom op with the flashqla backend
    selected, verifying that core_attn_out is filled in-place."""
    torch.manual_seed(1)
    num_seqs = 3
    seq_lens = torch.randint(1, 130, (num_seqs,), dtype=torch.int32)
    cu_seqlens = torch.zeros(num_seqs + 1, device="cuda", dtype=torch.int32)
    cu_seqlens[1:] = seq_lens.to(device="cuda").cumsum(0)
    total_tokens = int(cu_seqlens[-1].item())

    num_k_heads, num_v_heads = 4, 8
    head_k_dim = head_v_dim = 128
    dtype = torch.bfloat16
    q = F.normalize(
        torch.randn(
            1, total_tokens, num_k_heads, head_k_dim, device="cuda", dtype=dtype
        ).float(),
        p=2,
        dim=-1,
    ).to(dtype)
    k = F.normalize(
        torch.randn(
            1, total_tokens, num_k_heads, head_k_dim, device="cuda", dtype=dtype
        ).float(),
        p=2,
        dim=-1,
    ).to(dtype)
    v = torch.randn(
        1, total_tokens, num_v_heads, head_v_dim, device="cuda", dtype=dtype
    )
    a = torch.randn(1, total_tokens, num_v_heads, device="cuda", dtype=dtype)
    b = torch.randn(1, total_tokens, num_v_heads, device="cuda", dtype=dtype)
    A = torch.empty(num_v_heads, device="cuda", dtype=torch.float32).uniform_(0, 16)
    A_log = torch.log(A)
    dt = torch.exp(
        torch.rand(num_v_heads, device="cuda", dtype=torch.float32)
        * (math.log(0.1) - math.log(0.001))
        + math.log(0.001)
    )
    dt = torch.clamp(dt, min=1e-4)
    dt_bias = dt + torch.log(-torch.expm1(-dt))
    g = -A_log.exp().view(1, 1, num_v_heads) * F.softplus(
        a.float() + dt_bias.view(1, 1, num_v_heads)
    )
    beta = torch.sigmoid(b.float())
    initial_state = (
        torch.randn(num_seqs, num_v_heads, head_v_dim, head_k_dim, device="cuda") * 0.05
    )

    cfg = create_vllm_config(
        model_name="Qwen/Qwen3.5-0.8B",
        block_size=16,
        hf_config_override={"linear_key_head_dim": head_k_dim},
    )
    cfg.additional_config = {"gdn_prefill_backend": "flashqla"}
    with set_current_vllm_config(cfg):
        op = ChunkGatedDeltaRule()
        assert op.gdn_prefill_backend == "flashqla"
        core_attn_out = torch.empty(
            total_tokens, num_v_heads, head_v_dim, device="cuda", dtype=dtype
        )
        with torch.no_grad():
            actual_o, _ = op(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=False,
                core_attn_out=core_attn_out,
            )
    torch.accelerator.synchronize()

    # core_attn_out must be a copy of the returned output (bf16).
    core_err = (core_attn_out.float() - actual_o.squeeze(0).float()).abs()
    assert core_err.max().item() == 0
