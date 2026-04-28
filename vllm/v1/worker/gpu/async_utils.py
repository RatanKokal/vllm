# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# COPY-ONCE-AT-END MODIFICATION (v0.15.1)
# =========================================
# Original design: AsyncOutput.__init__() immediately kicked off async
# GPU->CPU copies for sampled_token_ids, logprobs_tensors, num_nans, and
# num_sampled_tokens the instant the object was constructed (i.e., every
# decode step).  get_output() then synchronised on copy_event and called
# .tolist() / .numpy().
#
# New design: __init__ records GPU tensor *references* only.  No copy
# stream, no copy_event, no non-blocking .to("cpu") call.  A single
# blocking copy happens inside get_output() exactly once, when the caller
# (scheduler / API layer) actually needs the data.  This eliminates the
# per-step copy-stream overhead and all intermediate host-visibility
# synchronisations on the hot path.
#
# Files that called async_barrier() relied on the per-step copy_event to
# gate stream scheduling.  With one-shot copies there is nothing to
# barrier against between steps, so async_barrier now degenerates to a
# no-op context manager (kept for API compatibility).

from contextlib import contextmanager

import numpy as np
import torch

from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    LogprobsTensors,
    ModelRunnerOutput,
)
from vllm.v1.worker.gpu.sample.output import SamplerOutput


class AsyncOutput(AsyncModelRunnerOutput):
    """Deferred-copy async output.

    On construction we keep GPU tensor references.  The actual blocking
    Device->Host transfer happens once inside get_output(), eliminating
    the per-step copy stream and associated synchronisation overhead.
    """

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampler_output: SamplerOutput,
        num_sampled_tokens: torch.Tensor,
        # copy_stream / copy_event kept as parameters so call-sites need
        # no changes, but they are intentionally *not used* here.
        copy_stream: torch.cuda.Stream,
        copy_event: torch.cuda.Event,
    ):
        # ------------------------------------------------------------------ #
        # Keep GPU tensor references.  No copy is initiated here.
        # We deliberately ignore copy_stream and copy_event – there is no
        # async transfer in flight, so there is nothing to synchronise on
        # between steps.
        # ------------------------------------------------------------------ #
        self.model_runner_output = model_runner_output

        # Retain strong references so CUDA doesn't free the tensors while
        # they are still live on-device.
        self._sampled_token_ids_gpu: torch.Tensor = (
            sampler_output.sampled_token_ids
        )
        self._logprobs_tensors_gpu: LogprobsTensors | None = (
            sampler_output.logprobs_tensors
        )
        self._num_nans_gpu: torch.Tensor | None = sampler_output.num_nans
        self._num_sampled_tokens_gpu: torch.Tensor = num_sampled_tokens

        # Prompt logprobs live in model_runner_output.prompt_logprobs_dict
        # already as LogprobsTensors (GPU); we snapshot that dict now.
        self._prompt_logprobs_dict_gpu: dict[str, LogprobsTensors | None] = (
            dict(model_runner_output.prompt_logprobs_dict)
        )

    # ---------------------------------------------------------------------- #
    # Public interface used by input_batch.set_async_sampled_token_ids().
    # The old AsyncGPUModelRunnerOutput exposed .sampled_token_ids_cpu and
    # .async_copy_ready_event so that the *next* step's input-prep could
    # consume the previous step's tokens without a full sync.
    #
    # In the copy-once design, the next step's input-prep reads from the
    # GPU tensor directly (prev_sampled_token_ids), so we only need to
    # expose a dummy event and the GPU tensor itself.
    # ---------------------------------------------------------------------- #
    @property
    def sampled_token_ids_cpu(self) -> torch.Tensor:
        """CPU copy – produced lazily on first access (blocking)."""
        if not hasattr(self, "_sampled_token_ids_cpu_cache"):
            # Single blocking copy.  Called at most once.
            self._sampled_token_ids_cpu_cache = (
                self._sampled_token_ids_gpu.to("cpu")
            )
        return self._sampled_token_ids_cpu_cache

    @property
    def async_copy_ready_event(self) -> torch.cuda.Event:
        """Compatibility shim – returns an already-recorded event."""
        if not hasattr(self, "_dummy_event"):
            self._dummy_event = torch.cuda.Event()
            # Record immediately: the data is already on CPU (or will be
            # produced synchronously by .sampled_token_ids_cpu above).
            self._dummy_event.record()
        return self._dummy_event

    # ---------------------------------------------------------------------- #
    # Core method: single blocking Device->Host copy, called once.
    # ---------------------------------------------------------------------- #
    def get_output(self) -> ModelRunnerOutput:
        """Perform the one-shot GPU->CPU copy and return ModelRunnerOutput.

        This is a blocking call.  It should be called exactly once per
        AsyncOutput instance, at the point the scheduler/API layer actually
        needs the data (i.e., when the request completes or a stream-flush
        interval fires).
        """
        # --- sampled token ids -------------------------------------------- #
        # Blocking copy: shape [num_reqs, max_gen_len]
        sampled_token_ids_cpu: torch.Tensor = self._sampled_token_ids_gpu.to("cpu")
        num_sampled_tokens_cpu: np.ndarray = (
            self._num_sampled_tokens_gpu.to("cpu").numpy()
        )

        max_gen_len: int = sampled_token_ids_cpu.shape[-1]
        num_reqs: int = sampled_token_ids_cpu.shape[0]

        if max_gen_len == 1:
            # Common non-spec-decode path.
            valid_sampled_token_ids: list[list[int]] = (
                sampled_token_ids_cpu.tolist()
            )
            logprobs_lists = None
            if self._logprobs_tensors_gpu is not None:
                # Blocking copy for logprobs.
                logprobs_cpu = LogprobsTensors(
                    self._logprobs_tensors_gpu.logprob_token_ids.to("cpu"),
                    self._logprobs_tensors_gpu.logprobs.to("cpu"),
                    self._logprobs_tensors_gpu.selected_token_ranks.to("cpu"),
                )
                logprobs_lists = logprobs_cpu.tolists()
        else:
            # Spec-decode path: use RejectionSampler parser.
            # Import here to avoid circular imports at module load time.
            from vllm.v1.worker.gpu.spec_decode.rejection_sampler import (  # noqa: E501
                RejectionSampler,
            )
            logprobs_gpu = self._logprobs_tensors_gpu
            logprobs_cpu_for_parse: LogprobsTensors | None = None
            if logprobs_gpu is not None:
                logprobs_cpu_for_parse = LogprobsTensors(
                    logprobs_gpu.logprob_token_ids.to("cpu"),
                    logprobs_gpu.logprobs.to("cpu"),
                    logprobs_gpu.selected_token_ranks.to("cpu"),
                )
            valid_sampled_token_ids, logprobs_lists = (
                RejectionSampler.parse_output(
                    sampled_token_ids_cpu,
                    self.model_runner_output.vocab_size
                    if hasattr(self.model_runner_output, "vocab_size")
                    else sampled_token_ids_cpu.shape[-1],
                    [],  # invalid_req_indices already applied during bookkeeping
                    logprobs_tensors=logprobs_cpu_for_parse,
                )
            )

        # --- num_nans ----------------------------------------------------- #
        if self._num_nans_gpu is not None:
            num_nans_list: list[int] = self._num_nans_gpu.to("cpu").tolist()
            self.model_runner_output.num_nans_in_logits = {
                req_id: num_nans_list[i]
                for i, req_id in enumerate(self.model_runner_output.req_ids)
            }

        # --- prompt logprobs ---------------------------------------------- #
        prompt_logprobs_dict: dict[str, LogprobsTensors | None] = {}
        for k, v in self._prompt_logprobs_dict_gpu.items():
            if v is not None:
                prompt_logprobs_dict[k] = LogprobsTensors(
                    v.logprob_token_ids.to("cpu"),
                    v.logprobs.to("cpu"),
                    v.selected_token_ranks.to("cpu"),
                )
            else:
                prompt_logprobs_dict[k] = None

        # --- materialise output ------------------------------------------- #
        output = self.model_runner_output
        output.sampled_token_ids = valid_sampled_token_ids
        output.logprobs = logprobs_lists
        output.prompt_logprobs_dict = prompt_logprobs_dict

        # Release GPU tensor references so their memory can be reclaimed.
        del self._sampled_token_ids_gpu
        del self._logprobs_tensors_gpu
        del self._num_nans_gpu
        del self._num_sampled_tokens_gpu
        del self._prompt_logprobs_dict_gpu

        return output


@contextmanager
def async_barrier(event: torch.cuda.Event | None):
    """Stream-coordination barrier.

    In the original implementation this synchronised on the per-step
    copy_event before yielding and re-recorded it after.  With copy-once
    semantics there is no per-step copy in flight, so this degenerates to
    a plain context manager.  Kept for API compatibility with call-sites
    that wrap draft-token or input-prep work in async_barrier().
    """
    # No synchronisation needed: no async copy is in flight between steps.
    try:
        yield
    finally:
        pass


def async_copy_to_np(x: torch.Tensor) -> np.ndarray:
    """Utility retained for call-sites that still use it.

    In the copy-once design this is called only from get_output(), not
    from the per-step hot path.
    """
    return x.to("cpu", non_blocking=True).numpy()
