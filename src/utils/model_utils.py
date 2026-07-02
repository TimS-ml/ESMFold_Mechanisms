"""
ESMFold Model Loading
======================

Shared helper for loading facebook/esmfold_v1 + tokenizer with a configurable
precision mode. The paper's original code loads in full fp32, which uses
~14.3GB VRAM for a ~100-residue sequence -- too tight on a 16GB GPU to leave
headroom for the activation patching / hook-based experiments in this repo
(donor+acceptor forward passes, stored intermediate representations, probe
training, etc).

Only `model.esm` (the ESM-2 language-model backbone, ~3B of the model's ~3B+
params) is put in reduced precision. `model.trunk` (folding trunk + structure
module) and all output heads stay fp32. This is deliberate, not a shortcut:

  - `modeling_esmfold.py` casts the ESM backbone's output back to fp32 right
    after it exits `model.esm` (`esm_s = esm_s.to(self.esm_s_combine.dtype)`,
    and `esm_s_combine` is a top-level fp32 param). So everything downstream
    of the backbone -- trunk blocks, s/z representations, plddt, ptm, etc --
    stays fp32 regardless of the backbone's precision.
  - This matters because this repo calls `.numpy()` directly on hook-captured
    s/z/plddt tensors all over the place (sklearn probe training, plotting,
    AUC scoring). bf16/fp16 tensors CANNOT be converted with `.numpy()`
    (`TypeError: Got unsupported ScalarType BFloat16`) -- verified empirically
    (see dtype_leak_test in the setup notes). Reducing precision on the whole
    model (`model.half()` / `model.bfloat16()`) breaks those call sites.
    `model.half()` additionally crashes outright inside `compute_tm` (PTM
    head) because fp16's narrow exponent range overflows the ptm logits.

Benchmarked on an RTX 4090 Laptop (16GB) with a 102-residue sequence
(fp32 baseline: 14.4GB peak, mean pLDDT 0.6148):

    precision    peak VRAM   mean pLDDT (delta)   extra deps
    fp32         14.4 GB     0.6148 (baseline)    --
    esm_half      8.6 GB     0.6145 (-0.0003)     --
    esm_bf16      8.6 GB     0.6164 (+0.0016)     --
    esm_fp8       6.0 GB     0.6139 (-0.0009)     torchao

Precision modes:
    "esm_fp8"  (default) -- torchao Float8WeightOnlyConfig on model.esm's
                Linear layers. Best VRAM savings and best fidelity to fp32
                of the reduced modes; weight-only quantization keeps ALL
                activations (including hook-captured ones) in fp32, so it's
                the safest choice for this codebase. No speedup on
                torch < 2.11 (no fp8 tensor-core kernel yet; pure storage
                savings). Requires `pip install torchao`.
    "esm_bf16" -- model.esm.bfloat16(). No extra deps. Slightly more VRAM
                than fp8. Note: raw ESMEncoderHooks captures (used by
                module_patching.py's encoder-level patching) will be bf16
                under this mode -- fine for tensor-to-tensor patching, but
                would break if any code called .numpy() on them directly
                (nothing currently does).
    "esm_half" -- model.esm.half(). HF's originally documented recipe.
                Same VRAM as esm_bf16; fp16 has a narrower dynamic range.
    "fp32"     -- no conversion. Matches the paper's original setup exactly.

Choosing the default automatically (no config needed)
------------------------------------------------------
fp8 tensor-core support is only present on Ada Lovelace / Hopper+ GPUs
(NVIDIA compute capability >= 8.9, e.g. RTX 40-series, L4, L40, H100). On
older GPUs (Ampere/RTX 30-series, V100, ...) torchao's fp8 kernels are
missing or unreliable. So instead of hardcoding "esm_fp8" as the default,
`load_esmfold()` auto-detects at call time:

    1. Explicit `precision=...` argument            -> always respected as-is
    2. $ESMFOLD_PRECISION environment variable       -> always respected as-is
    3. Otherwise, auto-select based on hardware:
         - CUDA device with compute capability >= 8.9 AND torchao importable
               -> "esm_fp8"
         - anything else (older GPU, CPU, or torchao not installed)
               -> "esm_bf16"  (works everywhere, no extra deps)

Cases 1 and 2 are explicit user intent, so an unavailable precision (e.g.
torchao missing) raises a clear ImportError. Case 3 is a silent fallback
choice, so it never errors -- it just prints which mode it picked and why.

Override the default globally via the ESMFOLD_PRECISION environment variable
without touching any call sites, e.g.:
    ESMFOLD_PRECISION=fp32 python src/block_patching.py ...
"""
import os

import torch
from transformers import AutoTokenizer, EsmForProteinFolding

MODEL_NAME = "facebook/esmfold_v1"
VALID_PRECISIONS = ("esm_fp8", "esm_bf16", "esm_half", "fp32")

# Minimum (major, minor) CUDA compute capability with fp8 tensor-core support.
# 8.9 = Ada Lovelace (RTX 40-series, L4, L40); 9.0 = Hopper (H100). Ampere
# (8.x < 8.9, e.g. RTX 30-series / A100) and older do NOT have fp8 tensor
# cores -- torchao's fp8 kernels are unsupported/unreliable there.
MIN_FP8_COMPUTE_CAPABILITY = (8, 9)


def _torchao_available() -> bool:
    try:
        import torchao  # noqa: F401
        from torchao.quantization import quantize_, Float8WeightOnlyConfig  # noqa: F401
    except ImportError:
        return False
    return True


def _gpu_supports_fp8(device) -> bool:
    """Whether `device` is a CUDA GPU with fp8 tensor-core support (compute capability >= 8.9)."""
    if not torch.cuda.is_available():
        return False
    dev = torch.device(device)
    if dev.type != "cuda":
        return False
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(index)
    return capability >= MIN_FP8_COMPUTE_CAPABILITY


def resolve_precision(device, precision=None, verbose=True) -> str:
    """Resolve which precision mode to use, following the 3-step order documented
    in this module's docstring (explicit arg > env var > hardware auto-detect).
    """
    if precision is not None:
        return precision

    env_precision = os.environ.get("ESMFOLD_PRECISION")
    if env_precision:
        return env_precision

    # No explicit choice made -- auto-detect from hardware. This branch never
    # raises; it just picks the best available option and says why.
    if _gpu_supports_fp8(device) and _torchao_available():
        return "esm_fp8"

    if verbose:
        dev = torch.device(device)
        if not torch.cuda.is_available():
            reason = "no CUDA device available"
        elif dev.type != "cuda":
            reason = f"device={device!r} is not a CUDA device"
        elif not _gpu_supports_fp8(device):
            index = dev.index if dev.index is not None else torch.cuda.current_device()
            capability = torch.cuda.get_device_capability(index)
            reason = (
                f"GPU compute capability {capability} < {MIN_FP8_COMPUTE_CAPABILITY} "
                "(fp8 tensor cores need Ada Lovelace/Hopper+)"
            )
        else:
            reason = "torchao not installed"
        print(f"Auto-detect: falling back to precision='esm_bf16' ({reason}). "
              f"Override with precision=... or $ESMFOLD_PRECISION if needed.")
    return "esm_bf16"


def _apply_precision(model, precision: str):
    if precision == "fp32":
        return model
    if precision == "esm_half":
        model.esm = model.esm.half()
        return model
    if precision == "esm_bf16":
        model.esm = model.esm.bfloat16()
        return model
    if precision == "esm_fp8":
        try:
            from torchao.quantization import quantize_, Float8WeightOnlyConfig
        except ImportError as e:
            raise ImportError(
                "precision='esm_fp8' requires torchao (`pip install torchao`). "
                "Alternatively pass precision='esm_bf16' (no extra deps) or "
                "set ESMFOLD_PRECISION=esm_bf16."
            ) from e
        quantize_(model.esm, Float8WeightOnlyConfig())
        return model
    raise ValueError(f"Unknown precision {precision!r}, expected one of {VALID_PRECISIONS}")


def load_esmfold(device, precision=None, chunk_size=None, verbose=True):
    """Load facebook/esmfold_v1 + its tokenizer.

    Args:
        device: torch device (str or torch.device) to move the model to.
        precision: one of "esm_fp8", "esm_bf16", "esm_half", "fp32". If not
            given, falls back to the $ESMFOLD_PRECISION env var, then to a
            hardware auto-detect (fp8 only on compute capability >= 8.9 GPUs
            with torchao installed, else bf16). See module docstring.
        chunk_size: if given, calls model.trunk.set_chunk_size(chunk_size)
            (a speed/memory tradeoff for the triangular attention ops in the
            trunk; unrelated to numeric precision).
        verbose: print progress messages.

    Returns:
        (model, tokenizer)
    """
    precision = resolve_precision(device, precision, verbose=verbose)
    if precision not in VALID_PRECISIONS:
        raise ValueError(f"Unknown precision {precision!r}, expected one of {VALID_PRECISIONS}")

    if verbose:
        print(f"Loading ESMFold model ({MODEL_NAME}, precision={precision})...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmForProteinFolding.from_pretrained(MODEL_NAME)
    model = _apply_precision(model, precision)
    model = model.to(device)
    model.eval()
    if chunk_size is not None:
        model.trunk.set_chunk_size(chunk_size)

    if verbose:
        print("Model loaded")
    return model, tokenizer
