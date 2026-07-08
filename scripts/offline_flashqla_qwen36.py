# SPDX-License-Identifier: Apache-2.0
"""Offline inference smoke test for FlashQLA GDN prefill backend on
Qwen3.6-35B-A3B-NVFP4 (Qwen3.5 MoE, NVFP4 quantized).

Run with CUDA_HOME=/usr/local/cuda and CUDA_VISIBLE_DEVICES limited to the
free GPUs. Verifies the flashqla backend is actually selected and that the
model produces a sensible text completion.
"""

import os

from vllm import LLM, SamplingParams

MODEL = "/ssd/nfs/models/Qwen/Qwen3.6-35B-A3B-NVFP4"


def main():
    llm = LLM(
        model=MODEL,
        tensor_parallel_size=int(os.environ.get("TP_SIZE", "4")),
        max_model_len=8192,
        max_num_seqs=4,
        enforce_eager=False,
        trust_remote_code=True,
        # Request the FlashQLA GDN prefill backend. Backend selection is logged
        # per-layer ("Using FlashQLA GDN prefill kernel ...") by
        # _log_gdn_backend_decision during model load.
        additional_config={"gdn_prefill_backend": "flashqla"},
    )

    prompts = [
        "Give me a short introduction to large language models.",
        "What is the capital of France? Answer in one word.",
    ]
    sampling = SamplingParams(temperature=0.0, max_tokens=128)
    outputs = llm.generate(prompts, sampling)
    for i, out in enumerate(outputs):
        text = out.outputs[0].text
        print(f"\n--- Prompt {i} ---\n{out.prompt}\n--- Response ---\n{text}")


if __name__ == "__main__":
    main()
