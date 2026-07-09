#!/usr/bin/env python
"""
inspect_model.py -- load ESMFold and print its module structure + tensor dims,
so we can build a /llm-style 3D computation-graph visualization.

The folding trunk block is a transformer encoder (LayerNorm + attention + MLP)
coupled to a pair-tensor track. This prints the real submodules and shapes.

Usage:
    conda activate protein_folding
    PYTHONPATH=src python scripts/inspect_model.py
"""
import sys
from pathlib import Path

import torch  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from transformers import EsmForProteinFolding  # noqa: E402


def shapes(module, prefix=""):
    """Print each submodule; for leaves show weight/feature dims."""
    for name, child in module.named_children():
        leaf = len(list(child.children())) == 0
        desc = child.__class__.__name__
        extra = ""
        if hasattr(child, "in_features"):
            extra = f"({child.in_features} -> {child.out_features})"
        elif hasattr(child, "normalized_shape"):
            extra = f"{tuple(child.normalized_shape)}"
        elif hasattr(child, "weight") and hasattr(child.weight, "shape"):
            extra = f"weight{tuple(child.weight.shape)}"
        print(f"{prefix}{name}: {desc} {extra}")
        if not leaf:
            shapes(child, prefix + "    ")


def main():
    print("loading facebook/esmfold_v1 on CPU (structure only, no forward)...")
    model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")
    model.eval()

    cfg = model.config
    print("\n================ CONFIG ================")
    ec = model.esm.config
    print(f"ESM-2:  layers={ec.num_hidden_layers}  hidden={ec.hidden_size}  "
          f"heads={ec.num_attention_heads}  ffn={ec.intermediate_size}")
    try:
        tc = cfg.esmfold_config.trunk
        print(f"Trunk:  blocks={tc.num_blocks}  seq_dim={tc.sequence_state_dim}  "
              f"pair_dim={tc.pairwise_state_dim}  seq_heads={tc.sequence_head_width}  "
              f"pair_heads={tc.pairwise_head_width}")
        sm = tc.structure_module
        print(f"StructMod: blocks={sm.num_blocks}  c_s={sm.sequence_dim}  c_z={sm.pairwise_dim}  "
              f"ipa_heads={sm.num_heads_ipa}")
    except Exception as e:
        print("(config introspection partial):", e)

    print("\n================ ESM-2 encoder layer[0] ================")
    shapes(model.esm.encoder.layer[0])

    print("\n================ Trunk block[0] (EsmFoldTriangularSelfAttentionBlock) ================")
    print(f"n trunk blocks = {len(model.trunk.blocks)}")
    shapes(model.trunk.blocks[0])

    print("\n================ Structure module ================")
    shapes(model.trunk.structure_module)

    print("\n================ Top-level EsmForProteinFolding children ================")
    for name, child in model.named_children():
        n = sum(p.numel() for p in child.parameters())
        print(f"{name}: {child.__class__.__name__}  ({n/1e6:.1f}M params)")


if __name__ == "__main__":
    main()
