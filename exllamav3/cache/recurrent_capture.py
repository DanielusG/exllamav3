"""
The recurrent state of one sequence at a page boundary inside a prompt chunk (fork:
Generator.recurrent_checkpoint_in_pass).

A recurrent checkpoint is a copy of the state at a page boundary, and the state is whole only
where a pass ends, so the library ends a pass at every position where it takes one: a prompt's
last page boundary gets a pass of its own for the tokens past it. With a capture the chunk is
not cut there: the job puts a RecurrentCapture in the chunk's params, each module with recurrent
state copies its state at that position while the pass goes on (the GDN scan, which runs in
sub-ranges anyway, stops there for a moment; a PLE layer ends a row slab there), and the
chunk's record stashes the copies as the checkpoint, in the format a regular stash has
(GDNState.stash, PLELayerState.stash: the same tensors, shapes and dtypes), which the
checkpoint cache and the disk tier take as they are.

A module that holds recurrent state and takes part says so with `captures_recurrent_state =
True` on its class (Generator checks every recurrent module of the model before turning the
feature on). In its forward:

    cap = params.get("recurrent_capture")
    if cap is not None and cap.wants(bsz, seqlen):
        ...
        cap.take(layer_state, (tensor, tensor))   # what layer_state.stash(slot) would return, on the device

The tensors are taken on the device the pass runs on, in the issuing thread's stream order;
the record copies them to the host after the chunk's passes have run (on the prefill thread
and in the pipeline the chunk's lane is synchronized before its record), as a regular stash
copies the live state.
"""

from __future__ import annotations
import torch
from .recurrent import host_copy


class RecurrentCapture:

    def __init__(self, offset: int, position: int):
        # Tokens from the start of the chunk to the boundary, and the boundary's position in the sequence
        self.offset = offset
        self.position = position
        # id(layer state) -> (layer state, tensors)
        self.layers = {}


    def wants(self, bsz: int, seqlen: int) -> bool:
        """Whether a pass of this shape takes a copy: one sequence, with the boundary strictly inside
        the pass (a boundary at the end of a pass is what the regular stash covers)"""
        return bsz == 1 and 0 < self.offset < seqlen


    def take(self, layer_state, tensors: tuple):
        self.layers[id(layer_state)] = (layer_state, tuple(tensors))


    def complete(self, cache) -> bool:
        """Every recurrent layer of the cache took its copy"""
        layers = cache.get_all_recurrent_layers()
        return all(id(l) in self.layers for l in layers.values())


    def state(self, cache, checkpoint_size: int) -> CapturedState | None:
        """What RecurrentCache.put takes for the captured state, None when a layer is missing"""
        return CapturedState(self, cache, checkpoint_size) if self.complete(cache) else None


class CapturedState:
    """
    A captured state as RecurrentCache.put sees a job's PairedState: checkpoint_size, and stash()
    in PairedState.stash's format with the target side only (a recurrent draft model's side is
    not captured: Generator.recurrent_checkpoint_in_pass is off with one). The host copies are
    made by stash(), i.e. only when the cache stores the checkpoint (not for a key it has).
    """

    def __init__(self, capture: RecurrentCapture, cache, checkpoint_size: int):
        self.capture = capture
        self.cache = cache
        self.checkpoint_size = checkpoint_size
        self.position = capture.position


    def stash(self) -> dict:
        target = {
            "position": self.position,
            "checkpoint_size": self.checkpoint_size,
        }
        for k, l in self.cache.get_all_recurrent_layers().items():
            _, tensors = self.capture.layers[id(l)]
            target[k] = tuple(host_copy(t) for t in tensors)
        return {
            "position": self.position,
            "checkpoint_size": self.checkpoint_size,
            "target": target,
        }
