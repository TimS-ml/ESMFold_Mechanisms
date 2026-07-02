"""
Representation Collection Utilities
====================================

Hook-based utilities for collecting intermediate representations from ESMFold.
Provides non-invasive collection of activations at any layer without modifying
the model's forward pass.

Classes:
    CollectedRepresentations: Container for collected activations (s, z, etc.)
    ESMEncoderHooks: Collects ESM language model layer outputs
    TrunkHooks: Collects folding trunk block outputs (s, z, seq2pair, pair2seq)
    IPAHooks: Collects structure module IPA outputs
"""

import argparse
import os
import types
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from contextlib import contextmanager

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm

from transformers import EsmForProteinFolding, AutoTokenizer
from transformers.models.esm.modeling_esmfold import (
    categorical_lddt,
    EsmFoldingTrunk,
    EsmForProteinFoldingOutput,
)
from transformers.models.esm.openfold_utils import (
    compute_predicted_aligned_error,
    compute_tm,
    make_atom14_masks,
    Rigid,
    Rotation,
)
from transformers.utils import ContextManagers

@dataclass
class CollectedRepresentations:
    """Container for collected representations from ESMFold."""
    # default_factory=dict (rather than `= {}`) is required for mutable
    # dataclass field defaults -- using a bare `{}` would share the *same*
    # dict across every CollectedRepresentations instance.

    # ESM encoder layers
    # keyed by layer index (0-35 for the 36-layer ESM-2 backbone); each
    # tensor is (batch, seq_len, 2560).
    esm_layers: Dict[int, torch.Tensor] = field(default_factory=dict)
    
    # Trunk block outputs
    # keyed by trunk block index (0-47 for the 48-block folding trunk).
    # s: (batch, seq_len, c_s=1024); z: (batch, seq_len, seq_len, c_z=128).
    s_blocks: Dict[int, torch.Tensor] = field(default_factory=dict)
    z_blocks: Dict[int, torch.Tensor] = field(default_factory=dict)
    
    # Trunk intermediate representations (optional)
    # seq2pair_updates: the per-block update computed from s and added into
    # z, shape (batch, seq_len, seq_len, 128). pair2seq_biases: the per-block
    # attention bias computed from z and fed into s's self-attention, shape
    # (batch, seq_len, seq_len, num_heads).
    seq2pair_updates: Dict[int, torch.Tensor] = field(default_factory=dict)
    pair2seq_biases: Dict[int, torch.Tensor] = field(default_factory=dict)
    
    # Structure module IPA outputs
    # keyed by IPA *call* index (0-7 for the 8 structure-module blocks --
    # NOT the trunk block index), each tensor (batch, seq_len, c_s=384, the
    # structure module's own sequence dim).
    ipa_outputs: Dict[int, torch.Tensor] = field(default_factory=dict)
    
    def clear(self):
        """Clear all collected representations."""
        # __dataclass_fields__ holds the names of every field declared above;
        # clearing each dict in place (rather than reassigning the attributes)
        # keeps the same CollectedRepresentations instance reusable across runs.
        for attr in self.__dataclass_fields__:
            getattr(self, attr).clear()



# ============================================================================
# PART 2: HOOK MANAGERS (collection only, no forward patching)
# ============================================================================

class ESMEncoderHooks:
    """
    Collect ESM encoder layer outputs via hooks.
    
    Usage:
        collector = CollectedRepresentations()
        hooks = ESMEncoderHooks(model.esm, collector)
        hooks.register()
        outputs = model(**inputs)
        hooks.remove()
        # collector.esm_layers now populated
    """
    
    def __init__(self, esm_module: nn.Module, collector: CollectedRepresentations):
        """Store the target ESM module and shared collector; handle list starts empty."""
        self.esm = esm_module
        self.collector = collector
        # register_forward_hook() returns a RemovableHandle; we must keep a
        # reference to every handle or we lose the ability to unregister the
        # hook later (PyTorch itself does not track them for us).
        self.handles: List = []
    
    def register(self, layers: Any = 'all'):
        """Register hooks on ESM encoder layers."""
        if layers == 'all':
            # self.esm.encoder.layer is the ModuleList of 36 ESM-2 transformer
            # layers, so this is indices 0..35.
            layers = range(len(self.esm.encoder.layer))
        
        for idx in layers:
            # A *forward hook* is a callback PyTorch invokes automatically
            # right after a module's forward() returns (with the module,
            # its inputs, and its outputs), letting us observe activations
            # without touching the model's own forward-pass code at all.
            #
            # make_hook(layer_idx) exists to avoid a classic late-binding
            # closure bug: if `hook` closed over the loop variable `idx`
            # directly, every layer's hook would share the same `idx` cell,
            # and since hooks only *run* later (during the actual forward
            # pass, after this whole loop has already finished), they would
            # all see the loop's final value of `idx` and overwrite the same
            # dict key instead of each populating its own layer's entry.
            # Passing `idx` as a function argument freezes its current value
            # into a new `layer_idx` binding at each iteration.
            def make_hook(layer_idx):
                def hook(module, inputs, outputs):
                    # ESM layer outputs hidden states (possibly as tuple)
                    tensor = outputs[0] if isinstance(outputs, tuple) else outputs
                    # detach(): drop the autograd graph (we only need the
                    # values, not gradients); cpu(): free GPU memory since we
                    # may be capturing activations across all 36 layers.
                    self.collector.esm_layers[layer_idx] = tensor.detach().cpu()
                return hook
            
            handle = self.esm.encoder.layer[idx].register_forward_hook(make_hook(idx))
            self.handles.append(handle)
    
    def remove(self):
        """Remove all registered hooks."""
        # RemovableHandle.remove() unregisters the hook from its module so it
        # no longer fires on subsequent forward passes; clearing self.handles
        # keeps this manager's bookkeeping in sync and makes remove() safe to
        # call more than once.
        for h in self.handles:
            h.remove()
        self.handles.clear()
    
    def __enter__(self):
        """Enter the context manager: register hooks, then return self."""
        # Entering `with ESMEncoderHooks(...) as hooks:` registers the hooks
        # right before the block's forward pass runs.
        self.register()
        return self
    
    def __exit__(self, *args):
        """Exit the context manager: remove all hooks regardless of how the block exited."""
        # Runs when leaving the `with` block, whether normally or via an
        # exception (the args, i.e. exc_type/exc_val/exc_tb, are ignored so
        # hooks are removed either way) -- guarantees hooks never leak into
        # later, unrelated forward passes.
        self.remove()

class TrunkHooks:
    """Collect trunk block outputs via hooks."""
    
    def __init__(self, trunk: nn.Module, collector: CollectedRepresentations):
        """Store the target trunk module and shared collector; handle list starts empty."""
        self.trunk = trunk
        self.collector = collector
        self.handles: List = []
    
    def register(
        self,
        blocks: Any = 'all',
        collect_s: bool = True,
        collect_z: bool = True,
        collect_seq2pair: bool = False,  # NEW - defaults to False
        collect_pair2seq: bool = False,  # NEW - defaults to False
    ):
        """Register hooks on trunk blocks."""
        if blocks == 'all':
            # self.trunk.blocks is the ModuleList of 48 triangular
            # self-attention blocks that make up the folding trunk.
            blocks = range(len(self.trunk.blocks))
        
        for idx in blocks:
            block = self.trunk.blocks[idx]
            
            # Block output hook (s, z)
            if collect_s or collect_z:
                # Each trunk block's forward() always returns the plain
                # (sequence_state, pairwise_state) tuple directly (unlike the
                # ESM encoder layers, no isinstance check is needed here).
                # `block_idx` must be captured via this factory function (not
                # closed over the loop variable `idx` directly) for the same
                # late-binding reason explained in ESMEncoderHooks above --
                # otherwise every block's hook would collapse onto the same
                # (final) idx value.
                def make_block_hook(block_idx, do_s, do_z):
                    def hook(module, inputs, outputs):
                        s, z = outputs
                        if do_s:
                            self.collector.s_blocks[block_idx] = s.detach().cpu()
                        if do_z:
                            self.collector.z_blocks[block_idx] = z.detach().cpu()
                    return hook
                
                handle = block.register_forward_hook(make_block_hook(idx, collect_s, collect_z))
                self.handles.append(handle)
            
            # seq2pair hook
            if collect_seq2pair:
                # sequence_to_pair's forward returns a single tensor (the
                # per-block update derived from s that gets added into z),
                # not a tuple, hence no unpacking here.
                def make_s2p_hook(block_idx):
                    def hook(module, inputs, outputs):
                        self.collector.seq2pair_updates[block_idx] = outputs.detach().cpu()
                    return hook
                
                handle = block.sequence_to_pair.register_forward_hook(make_s2p_hook(idx))
                self.handles.append(handle)
            
            # pair2seq hook
            if collect_pair2seq:
                # pair_to_sequence's forward likewise returns a single tensor:
                # the attention bias derived from z that's fed into this
                # block's sequence self-attention.
                def make_p2s_hook(block_idx):
                    def hook(module, inputs, outputs):
                        self.collector.pair2seq_biases[block_idx] = outputs.detach().cpu()
                    return hook
                
                handle = block.pair_to_sequence.register_forward_hook(make_p2s_hook(idx))
                self.handles.append(handle)
    
    def remove(self):
        """Remove all registered hooks."""
        # Same lifecycle as ESMEncoderHooks.remove(): unregister every stored
        # handle, then clear the list so bookkeeping matches reality.
        for h in self.handles:
            h.remove()
        self.handles.clear()
    
    def __enter__(self):
        """Enter the context manager: register hooks, then return self."""
        # Registers hooks on entry to a `with TrunkHooks(...) as hooks:` block.
        self.register()
        return self
    
    def __exit__(self, *args):
        """Exit the context manager: remove all hooks regardless of how the block exited."""
        # Always removes hooks on exit (normal or exception), so they don't
        # keep firing on later forward passes.
        self.remove()


class IPAHooks:
    """
    Collect IPA outputs via hooks.
    
    Note: IPA is called multiple times per forward (once per SM block).
    Call reset() before each forward pass.
    
    Usage:
        collector = CollectedRepresentations()
        hooks = IPAHooks(model.trunk.structure_module, collector)
        hooks.register()
        hooks.reset()  # Important!
        outputs = model(**inputs)
        hooks.remove()
        # collector.ipa_outputs now populated
    """
    
    def __init__(self, structure_module: nn.Module, collector: CollectedRepresentations):
        """Store the target structure module and shared collector; handles/counter start empty/zero."""
        self.sm = structure_module
        self.collector = collector
        self.handles: List = []
        # Unlike ESMEncoderHooks/TrunkHooks (one hook per distinct submodule
        # instance, so the block index is fixed at registration time), there
        # is only *one* `ipa` submodule, and its forward() is called 8 times
        # per structure-module forward pass (once per structure-module
        # block). A single hook on that one module fires on every one of
        # those calls, so we need our own counter to tell them apart.
        self._call_idx = 0
    
    def register(self):
        """Register hook on IPA module."""
        def hook(module, inputs, outputs):
            # IPA's forward returns a single tensor (batch, seq_len, c_s=384)
            # that gets added residually into the structure module's running
            # sequence state -- not a tuple, so no unpacking needed.
            self.collector.ipa_outputs[self._call_idx] = outputs.detach().cpu()
            self._call_idx += 1
        
        handle = self.sm.ipa.register_forward_hook(hook)
        self.handles.append(handle)
    
    def reset(self):
        """Reset call counter. Call before each forward pass."""
        # _call_idx persists across forward passes; without calling this
        # first, a second forward pass would keep counting up from 8 instead
        # of restarting at 0, so ipa_outputs would grow instead of being
        # overwritten with each new forward pass's 8 fresh values.
        self._call_idx = 0
    
    def remove(self):
        """Remove all registered hooks."""
        # NOTE: possible bug -- unlike ESMEncoderHooks/TrunkHooks, this class
        # defines no __enter__/__exit__, so it can't be used as a context
        # manager despite having the same register()/remove() API; likely
        # just an oversight (use register() + reset() manually, as shown in
        # the class docstring's usage example).
        for h in self.handles:
            h.remove()
        self.handles.clear()