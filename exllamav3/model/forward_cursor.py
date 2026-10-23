from __future__ import annotations
import torch
from ..cache.recurrent_util import advance_recurrent_states

"""
A layer-split forward pass issued in pieces.

Model.forward() enqueues the whole pass and returns; with the layers of a model split over two
GPUs the second card then idles while the first one works and vice versa. A ForwardCursor runs
the same modules in the same order with the same params, but lets the caller stop between two
modules, so the generator can issue the second-card half of one batch and the first-card half of
another one alternately (see Generator.stagger_decode).

Rules a caller has to keep:

  - Two cursors may be advanced alternately only while they are on DIFFERENT devices. Every
    workspace the modules use is per device and relies on stream order (a site's output is
    consumed before the next site on that device runs), which holds for kernels of two passes on
    two devices and would not for two passes on the same one.
  - A cursor that is left waiting between two modules must hold() first: the stream stack may
    live in memory another pass on that device is entitled to reuse.
  - finish() advances the recurrent states, exactly once, after the last module.
  - What a pass leaves in a layer's state for after the pass is the caller's to keep: a
    speculative pass through recurrent layers (GDN, Mamba2) stages its scan inputs in buffers
    the layer's state holds once for every pass, and the rewind after verification replays
    them (see Generator._stagger_hold_spec).

Differences from Model.forward, all for a pass that waits between two modules: the CPU expert
workers are woken again when the pass resumes (wake()), the stream stack is copied when the pass
is left waiting (hold()), and a module output in pinned host memory is copied into pinned memory
of the pass's own (run_to()).
"""


def _pinned_host(x) -> bool:
    return torch.is_tensor(x) and x.device.type == "cpu" and x.is_pinned()


def _own_pinned(x: torch.Tensor) -> torch.Tensor:
    """A copy of x in pinned memory from torch's caching host allocator, which does not hand the
    block out again before the uploads read from it have run"""
    return torch.empty_like(x, pin_memory = True).copy_(x)


def cuda_stage_bounds(model) -> list[int]:
    """
    Indices into model.fwd_modules at which the pass moves on to another CUDA device (the
    first module of each later stage). Modules without a CUDA device of their own belong to
    the stage around them. Empty for a model on a single card.
    """
    bounds = []
    last = None
    for i, (module, _, _) in enumerate(model.fwd_modules):
        dev = getattr(module, "device", None)
        if dev is None:
            continue
        dev = torch.device(dev)
        if dev.type != "cuda":
            continue
        if last is not None and dev.index != last:
            bounds.append(i)
        last = dev.index
    return bounds


class ForwardCursor:

    def __init__(self, model, input_ids: torch.Tensor, params: dict):
        assert not model.loaded_tp, "ForwardCursor is for layer-split models"
        self.model = model
        self.input_ids = input_ids
        self.params = params
        self.pos = 0
        self.end = len(model.fwd_modules)
        self.finished = False
        self.hosts = list(getattr(model.config, "moe_cpu_hosts", {}).values())

        # Same prologue as Model.forward() / forward_ls()
        self.x = model.prepare_inputs(input_ids, params)
        for m in model._get_prefetch_layers:
            m.prefetch(self.x, params)
        self.wake()

    def wake(self):
        """Tell the CPU expert workers that jobs are coming (begin_pass). Called again when a
        waiting pass resumes, so the worker spins instead of napping."""
        for h in self.hosts:
            h.begin_pass()

    def run_to(self, stop: int):
        """Issue the modules up to (not including) index stop."""
        model = self.model
        params = self.params
        x = self.x
        stop = min(stop, self.end)
        while self.pos < stop:
            module, instance, _ = model.fwd_modules[self.pos]
            params["layer_instance"] = instance
            if module.caps.get("logits_output") and (num := params.get("last_tokens_only")):
                x = x[..., -num:, :].contiguous()
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)
            if _pinned_host(x):
                # The CPU embedding stages its output through one pinned buffer per shape, safe
                # for passes with a host sync between them; the next group's pass can rewrite it
                # before this pass's upload from it has run
                x = _own_pinned(x)
            self.pos += 1
            # Kept current after every module: an exception leaves a consistent cursor
            self.x = x

    def hold(self):
        """Take ownership of the state before waiting between two modules."""
        if torch.is_tensor(self.x):
            self.x = self.x.clone()

    def finish(self) -> torch.Tensor:
        """Issue whatever is left, advance the recurrent states and return the output."""
        assert not self.finished, "ForwardCursor finished twice"
        self.run_to(self.end)
        advance_recurrent_states(self.input_ids, self.params, self.model)
        self.finished = True
        return self.x


class DirectForward:
    """
    Model.forward() behind the cursor's finish(): what the generator's decode step gets when the
    pass is not staggered (and always in tensor-parallel mode), so that path stays what it was.
    """

    def __init__(self, model, input_ids: torch.Tensor, params: dict):
        self.model = model
        self.input_ids = input_ids
        self.params = params

    def finish(self) -> torch.Tensor:
        return self.model.forward(input_ids = self.input_ids, params = self.params)
