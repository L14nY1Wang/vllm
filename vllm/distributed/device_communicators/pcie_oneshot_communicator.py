# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM wrapper around b12x PCIe oneshot custom allreduce.

Size-gated dispatch: messages ≤ crossover bytes → PCIe oneshot (fast for
small/latency-bound allreduces). Messages > crossover → fall through to
PyNCCL (NCCL Ring wins at bandwidth-bound large messages).

Crossover is resolved once at init time from VLLM_PCIE_ONESHOT_MAX_SIZE
(accepts "128KB", "1MB", "auto", or raw byte count). "auto" runs a one-time
microbenchmark to find the exact crossover on this topology.
"""

import torch
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.logger import init_logger

logger = init_logger(__name__)

_DEFAULT_CROSSOVER = 128 * 1024  # 128KB — empirically optimal for 4x RTX 5090

pcie_oneshot_available = False
try:
    from b12x.distributed.pcie_oneshot import (  # type: ignore
        PCIeOneshotAllReducePool,
        SUPPORTED_DTYPES,
        SUPPORTED_WORLD_SIZES,
        parse_pcie_oneshot_max_size,
    )
    pcie_oneshot_available = True
except ImportError:
    pass


class PCIeOneshotCommunicator:
    """Size-gated PCIe oneshot allreduce backend for the TP group."""

    def __init__(self, group: ProcessGroup, device: int | str | torch.device):
        self.disabled = True
        self.pool = None
        self.max_size = _DEFAULT_CROSSOVER

        if not pcie_oneshot_available:
            return
        if not torch.cuda.is_available():
            return

        self.group = group
        self.world_size = torch.distributed.get_world_size(group)
        self.rank = torch.distributed.get_rank(group)
        if self.world_size == 1 or self.world_size not in SUPPORTED_WORLD_SIZES:
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        try:
            self.pool = PCIeOneshotAllReducePool.from_process_group(
                process_group=group, device=device,
                max_input_bytes=8 * 1024 * 1024, max_size=8 * 1024 * 1024,
                single_channel=True,
            )
            self._channel = self.pool.for_stream()

            raw = envs.VLLM_PCIE_ONESHOT_MAX_SIZE
            parsed = parse_pcie_oneshot_max_size(raw)
            if parsed is not None:
                self.max_size = int(parsed)
            else:
                try:
                    self.max_size = self._channel.find_crossover_size(group)
                except Exception as e:
                    logger.warning("PCIe oneshot autotune failed (%s), using %d",
                                   e, _DEFAULT_CROSSOVER)
                    self.max_size = _DEFAULT_CROSSOVER
        except Exception as e:
            logger.warning("PCIe oneshot disabled: %s", e)
            return

        self.disabled = False
        logger.info("PCIe oneshot allreduce ENABLED (world_size=%d, crossover=%d bytes).",
                     self.world_size, self.max_size)

    def should_pcie_ar(self, input_: torch.Tensor) -> bool:
        if self.disabled or self.pool is None:
            return False
        if input_.device != self.device or input_.dtype not in SUPPORTED_DTYPES:
            return False
        nbytes = input_.numel() * input_.element_size()
        if nbytes == 0 or nbytes > self.max_size or nbytes % 16 != 0:
            return False
        return self._channel.should_allreduce(input_)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor | None:
        try:
            return self.pool.all_reduce(input_)
        except Exception as e:
            logger.warning_once("PCIe oneshot runtime failure (%s), disabling.", e)
            self.disabled = True
            return None

    def close(self) -> None:
        if self.pool is not None:
            try: self.pool.close()
            except Exception: pass
            self.pool = None
