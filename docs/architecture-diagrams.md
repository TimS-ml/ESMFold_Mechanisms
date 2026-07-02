# Architecture Diagrams

Visual reference for how this repository's code is organized and how the core
experiments call into each other and into ESMFold itself. See
[`../README.md`](../README.md) for setup and [`../src/README.md`](../src/README.md)
for a per-module description of what each script does.

All diagrams are [Mermaid](https://mermaid.js.org/); GitHub renders them
inline automatically.

## Table of contents

1. [Repository overview](#1-repository-overview)
2. [ESMFold model internals](#2-esmfold-model-internals)
3. [`reproduce.sh` pipeline (10 steps)](#3-reproducesh-pipeline-10-steps)
4. [Core method A: activation patching](#4-core-method-a-activation-patching)
5. [Core method B: representation steering](#5-core-method-b-representation-steering)
6. [Shared utilities call graph](#6-shared-utilities-call-graph)
7. [Model loading / precision auto-detection](#7-model-loading--precision-auto-detection)

---

## 1. Repository overview

How the top-level pieces fit together: demo notebooks are the fast path (no
dataset download needed), `reproduce.sh` is the full paper-reproduction path,
and both ultimately load ESMFold through the same `src/utils/model_utils.py`
helper.

```mermaid
flowchart TD
    subgraph entry["Entry points"]
        demo["demo_notebooks/*.ipynb\n(patching / charge steering / distance steering demos)"]
        repro["reproduce.sh\n(10-step full paper reproduction)"]
    end

    subgraph data["Data (downloaded from HuggingFace)"]
        hfhub[("HF Hub:\nkevinlu4588/ProteinFolding")]
        datadir["data/*.csv, *.parquet"]
        modelsdir["models/*.pt\n(pretrained charge directions, probes)"]
        resultsdir["results/\n(experiment outputs)"]
    end

    subgraph core["src/ core experiment code"]
        utils["src/utils/\nmodel_utils.py, representation_utils.py, trunk_utils.py"]
        exp["src/*.py, src/main_paper/*.py, src/appendix/*.py\n(one file per experiment)"]
    end

    subgraph scripts["scripts/"]
        dl["download_from_hf.py / upload_to_hf.py"]
        fix["fix_datasets_*.py\n(dataset construction/cleaning)"]
        render["render_*.py\n(py3Dmol -> PNG frames -> video)"]
    end

    esmfold[("facebook/esmfold_v1\n(HuggingFace Hub, ~17GB weights)")]

    dl <--> hfhub
    hfhub -.populates.-> datadir
    hfhub -.populates.-> resultsdir
    fix --> datadir

    demo --> utils
    repro --> exp
    exp --> utils
    utils --> esmfold

    exp --> datadir
    exp --> modelsdir
    exp --> resultsdir
    render --> resultsdir
```

---

## 2. ESMFold model internals

Background needed to read every other diagram in this doc. All experiments
intervene on one of these three stages. `s` = per-residue ("sequence") state,
`z` = per-residue-pair ("pairwise") state.

```mermaid
flowchart LR
    seq["input sequence\n(amino acid string)"]

    subgraph encoder["ESM-2 encoder (model.esm)\n36 transformer layers, 2560-dim"]
        esmlayers["layer 0 ... layer 35"]
    end

    combine["esm_s_combine\n(learned softmax-weighted sum\nover all 37 layer outputs incl. embeddings)"]
    mlp["esm_s_mlp\n(2560 -> 1024)"]

    subgraph trunk["Folding trunk (model.trunk)\n48 blocks, recycled up to 4x"]
        direction TB
        block["block i:\nseq_attention (s update)\ntri_mul_in / tri_mul_out (z update)\ntri_att_start / tri_att_end (z update)\nsequence_to_pair / pair_to_sequence\n(cross-stream coupling)"]
    end

    subgraph sm["Structure module (model.trunk.structure_module)\n8 IPA blocks, shared weights"]
        ipa["Invariant Point Attention\n(s, z -> rigid frames -> backbone update)"]
    end

    heads["Output heads:\ndistogram_head, ptm_head, lddt_head, lm_head"]
    coords["3D coordinates (atom14/atom37)\n+ plddt, ptm, predicted_aligned_error"]

    seq --> encoder
    esmlayers -->|"hidden_states\n(all 37 layers)"| combine
    combine -->|"esm_s: [B,L,37,2560] -> [B,L,2560]"| mlp
    mlp -->|"s_s_0: [B,L,1024]"| trunk
    trunk -->|"z_0: [B,L,L,128]\n(initialized to zeros)"| trunk
    trunk -->|"s, z (recycled)"| sm
    sm -->|"s, z"| heads
    sm --> coords
    heads --> coords
```

Key facts used throughout the experiment code:

- The ESM-2 encoder dominates parameter count (~3B of ~3B+ params) -- this is
  why `src/utils/model_utils.py` only reduces precision on `model.esm`, not
  the trunk/structure module (see [§7](#7-model-loading--precision-auto-detection)).
- `s` and `z` are what activation patching, hooks, and steering all target.
- The trunk is **recycled**: it runs its 48 blocks, then feeds its own output
  back in as the next recycle's input (up to `num_recycles` times, default 4),
  refining the structure iteratively.

---

## 3. `reproduce.sh` pipeline (10 steps)

Each step is a standalone script (`python src/....py --args`); this diagram
shows the actual data dependencies between steps, i.e. which step's output
file feeds which later step's input. Case counts (`N_MODULE`, `N_BLOCK`, ...)
are configured at the top of `reproduce.sh`.

```mermaid
flowchart TD
    csv1[("data/patching_dataset.csv")]
    csv2[("data/all_block_patching_results.parquet")]
    csv3[("data/single_block_patching_successes.csv")]
    csv4[("data/probing_train_test.csv")]
    csv5[("data/target_loops_dataset.csv")]

    s1["1. module_patching.py\npatch whole components\n(ESM encoder / trunk / structure module)"]
    s2["2. module_plotting.py"]
    s3["3. block_patching.py\npatch one trunk block at a time\n(finds blocks 0-10 matter most)"]
    s4["4. representation_tracking.py\ntrack s/z evolution across blocks"]
    s5["5. final_sliding_window_ablation.py\nablate contiguous block windows"]
    s6["6. charge_steering.py\ntrain DoM charge directions,\nthen steer (induce hairpins)"]
    s7["7. charge_repulsion.py\nreuse directions from step 6,\nsteer (disrupt hairpins)"]
    s8["8. contact_steering.py\ntrain Ridge distance probes,\nthen steer (induce contacts)"]
    s9a["9a. bias_analysis2.py\npair-to-sequence bias vs contact-map AUC"]
    s9b["9b. bias_plotting.py"]
    s9c["9c. bias_patching.py\npatch bias term, measure causal effect"]
    s10["10. z_scaling.py\nscale s vs z independently,\ngradient sensitivity analysis"]

    csv1 --> s1 --> |"all_block_patching_results.parquet"| s2
    csv2 --> s3 --> |"single_block_patching_results.parquet\n(built-in plotting on flush)"| csv3
    csv3 --> s4
    csv3 --> s5
    csv4 --> s6
    csv5 --> s6
    s6 -->|"charge_directions.pt"| s7
    csv3 --> s7
    csv4 --> s8
    csv3 --> s8
    csv2 --> s9a --> |"metrics.parquet, sequence_info.parquet"| s9b
    csv3 --> s9c
    csv3 --> s10

    classDef step fill:#2d5,stroke:#333,color:#000;
    class s1,s2,s3,s4,s5,s6,s7,s8,s9a,s9b,s9c,s10 step
```

All steps share one model-loading call (`load_esmfold`, see [§7](#7-model-loading--precision-auto-detection))
and write into `results/<step_name>/`.

---

## 4. Core method A: activation patching

The central causal-tracing technique (`module_patching.py`, `block_patching.py`,
and the patching parts of `bias_patching.py` / `representation_tracking.py`).
A **donor** sequence (known to form a hairpin) and an **acceptor** sequence
(helical, no hairpin) are each run through the model; the donor's activations
at a chosen location are spliced into the acceptor's forward pass to test
whether that location causally carries the hairpin-forming signal.

```mermaid
sequenceDiagram
    participant Donor as donor_sequence\n(has a hairpin)
    participant Model as ESMFold model
    participant Hooks as ESMEncoderHooks /\nTrunkHooks / IPAHooks
    participant Collector as CollectedRepresentations\n(s_blocks, z_blocks, esm_layers, ...)
    participant Acceptor as acceptor_sequence\n(helical, no hairpin)
    participant Patch as patch_esm_layers() /\npatch context manager
    participant Detect as detect_hairpins()\n(DSSP-based)

    Hooks->>Model: register forward hooks
    Donor->>Model: forward(donor_sequence)
    Model->>Hooks: hook fires per block/layer
    Hooks->>Collector: store .detach().cpu() activation
    Hooks->>Model: remove hooks

    Note over Patch,Model: Patch context manager registers a\nforward_pre_hook / forward_hook that\nOVERWRITES the target block's output\nwith the donor's captured activation\n(only for the chosen block/window/positions)

    Patch->>Model: enter patch context (donor activations loaded)
    Acceptor->>Model: forward(acceptor_sequence)
    Model->>Model: at patched block: output replaced\nwith donor activation (rest of forward\npass runs normally downstream)
    Patch->>Model: exit patch context (hook removed)

    Model->>Detect: predicted structure (PDB)
    Detect->>Detect: run mkdssp -> 3-state SS string\n-> scan for antiparallel E-C-E hairpin motif
    Detect-->>Patch: hairpin_found: bool

    Note over Detect: hairpin_found == True at this block/module\n=> that component causally carries the\nhairpin-forming signal for this donor/acceptor pair
```

`module_patching.py` patches whole components (swap all 36 ESM layers, or all
48 trunk blocks, or the whole structure module at once) to find *which
component* is necessary/sufficient. `block_patching.py` repeats this one trunk
block at a time to find *which of the 48 blocks* matter (the paper's headline
finding: blocks 0-10 have the strongest causal effect).

---

## 5. Core method B: representation steering

Instead of copying a donor's raw activations (method A), steering *adds a
learned direction vector* to `s` or `z` at chosen blocks/positions, then
checks the effect on the predicted structure. Two direction-learning methods
feed three steering experiments:

```mermaid
flowchart TD
    subgraph training["Direction / probe training"]
        dom["charge_dom_training.py\nDifference-of-Means (DoM):\nmean(s | positive-charge residues)\n- mean(s | negative-charge residues)\nper trunk block -> charge_directions.pt"]
        probe["z_probing_distance.py\nRidge regression probe:\nz -> predicted CA-CA distance\nper trunk block -> probe weight vectors\nused directly as steering directions"]
    end

    subgraph steer["Steering experiments"]
        cs["charge_steering.py\nadd complementary (opposite) charge\nsignal to s at cross-strand positions\n=> INDUCE hairpin formation"]
        cr["charge_repulsion.py\nadd same-charge (repulsive) signal\nto s of an EXISTING donor hairpin\n=> DISRUPT hairpin formation"]
        ct["contact_steering.py\nadd distance-probe direction to z\nat a target residue pair\n=> INDUCE a specific CA-CA contact"]
    end

    dom -->|"charge_directions.pt\n(s_directions, s_stds per block)"| cs
    cs -->|"charge_directions.pt\n(reused, not retrained)"| cr
    probe -->|"probe weight vectors\n(DistanceProbe per block)"| ct

    cs --> eval1["detect_hairpins() / compute_handedness\nH-bond count, strand geometry"]
    cr --> eval2["detect_hairpins()\ndistance increase, H-bond decrease"]
    ct --> eval3["compute_ca_distances()\ntarget vs. achieved Cbeta-Cbeta distance"]
```

All three steering scripts share the same intervention mechanism: monkey-patch
`model.trunk.blocks[i].forward` (via `types.MethodType`) to add the direction
vector to `s` or `z` right after the block computes it, for a chosen window of
blocks and a chosen window of sequence positions, scaled by a magnitude in
units of the direction's standard deviation.

---

## 6. Shared utilities call graph

`src/utils/` has no dependency on any experiment script (arrows only point
*into* it), so every experiment script can freely import from it.

```mermaid
flowchart LR
    subgraph utils["src/utils/"]
        mu["model_utils.py\nload_esmfold()"]
        ru["representation_utils.py\nCollectedRepresentations\nESMEncoderHooks / TrunkHooks / IPAHooks"]
        tu["trunk_utils.py\nrun_dssp_on_pdb() / detect_hairpins()\ncompute_handedness_from_structure()"]
    end

    subgraph consumers["Experiment scripts (19 call sites across 17 files)"]
        c1["module_patching.py\nblock_patching.py"]
        c2["charge_steering.py\ncharge_repulsion.py\ncharge_dom_training.py"]
        c3["contact_steering.py\nz_probing_distance.py"]
        c4["bias_patching.py\nbias_analysis2.py\nbias_plotting.py"]
        c5["representation_tracking.py\nfinal_sliding_window_ablation.py\nz_scaling.py / z_scale_gradient.py"]
        c6["appendix/*.py\n(probe_analysis_v2, charge_probing_binary2,\nblock_analysis3, ...)"]
    end

    c1 & c2 & c3 & c4 & c5 & c6 -->|"model, tokenizer = load_esmfold(device)"| mu
    c1 & c3 & c4 & c5 & c6 -->|"hooks = TrunkHooks(model.trunk, collector)"| ru
    c1 & c2 --> tu
```

Most experiment scripts additionally define their **own** local, simplified
copies of the hook classes (e.g. `module_patching.py` has its own
`ESMEncoderHooks`) rather than importing from `representation_utils.py`, to
keep each script's core collection logic self-contained -- see the code
comments in each file's "PART 2: HOOK MANAGERS" section for the specifics of
what each local copy collects.

---

## 7. Model loading / precision auto-detection

`load_esmfold()` (`src/utils/model_utils.py`) is the single entry point every
experiment script uses to load `facebook/esmfold_v1`. Only `model.esm` (the
ESM-2 backbone) is ever put in reduced precision -- the trunk and structure
module always stay fp32, because `modeling_esmfold.py` upcasts the backbone's
output back to fp32 immediately (`esm_s = esm_s.to(self.esm_s_combine.dtype)`),
so this keeps every hook-captured `s`/`z`/`plddt` tensor `.numpy()`-safe for
the sklearn/plotting code used throughout this repo.

```mermaid
flowchart TD
    start(["load_esmfold(device, precision=None)"]) --> explicit{"precision\narg given?"}
    explicit -->|yes| useexplicit["use it as-is\n(raises if e.g. torchao missing)"]
    explicit -->|no| envvar{"$ESMFOLD_PRECISION\nset?"}
    envvar -->|yes| useenv["use it as-is\n(raises if e.g. torchao missing)"]
    envvar -->|no| autodetect["auto-detect (never raises):"]

    autodetect --> cc{"CUDA device with\ncompute capability >= 8.9?\n(Ada Lovelace / Hopper+)"}
    cc -->|no| bf16fallback["esm_bf16\n(works everywhere, no extra deps)"]
    cc -->|yes| tao{"torchao\nimportable?"}
    tao -->|no| bf16fallback
    tao -->|yes| fp8["esm_fp8\n(torchao Float8WeightOnlyConfig\non model.esm's Linear layers)"]

    useexplicit --> apply["_apply_precision(model, precision)"]
    useenv --> apply
    bf16fallback --> apply
    fp8 --> apply

    apply --> load["model.to(device).eval()\ntokenizer = AutoTokenizer.from_pretrained(...)"]
    load --> done(["return model, tokenizer"])
```

Benchmarked on an RTX 4090 Laptop (16GB VRAM), 102-residue sequence, fp32
baseline = 14.4GB peak / mean pLDDT 0.6148:

| precision | peak VRAM | mean pLDDT (delta) | extra deps |
|---|---|---|---|
| `fp32` | 14.4 GB | 0.6148 (baseline) | -- |
| `esm_half` | 8.6 GB | 0.6145 (-0.0003) | -- |
| `esm_bf16` | 8.6 GB | 0.6164 (+0.0016) | -- |
| `esm_fp8` | 6.0 GB | 0.6139 (-0.0009) | `torchao` |

See the `src/utils/model_utils.py` module docstring for the full rationale
(including why whole-model `bf16`/`half` conversion was rejected -- it breaks
`.numpy()` on hook-captured tensors and, for `half`, crashes `compute_tm`
outright from fp16 overflow).
