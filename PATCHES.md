# PATCHES.md — wbic16/exo divergence from exo-explore/exo

Fork-patch discipline: this file is the complete per-file record of how this
fork diverges from upstream, so the patch set stays re-appliable as upstream
moves. Base: exo-explore/exo main @ a60b9569 (exo 0.3.70), 2026-11-22.
Working branch: `amd-ubuntu-inkling`.

The patch set has two halves (Will, 2026-11-22):
(a) AMD Ubuntu node support — exo serving on Ubuntu Linux nodes with the
    AMD Radeon 780M (RDNA3 iGPU, unified DDR5), x86_64.
(b) Inkling-2-Mini model support — the architecture the ranch fleet serves
    (fleet artifacts are GGUF; conversion path documented per patch).

## Recovered patches (provenance: Will's February 2026 running)

### P0 — bf16 cast on the CLANG backend (fix-issue-458) — SUPERSEDED BY REWRITE
- Source: wbic16/exo branch `fix-issue-458`, commit 89815b16b25f35c2f376547130aaadd51d1eb914
  (Will Bickford, 2024-12-06), the branch's only commit over upstream cbcd952e.
- Upstream issue: exo-explore/exo#458, "clang issues on ubuntu 24.04 and
  Python 3.12" — exo failed at prompt time on Ubuntu 24.04 / Ryzen.
- File: `exo/inference/tinygrad/models/llama.py` (+6 lines) — in
  `fix_bf16()`, when `Device.DEFAULT == "CLANG"`, cast bfloat16 weights to
  float32 via `llvm_bf16_cast` instead of the float16 cast used elsewhere.
- Status on current upstream: the tinygrad backend was removed in exo's 2025
  rewrite (current backend is MLX); the file no longer exists, so the patch
  cannot re-apply literally. Its intent — on the CPU backend, bf16 weights
  must not take the fp16 path — is the seed of half (a)'s current work.
  Carried here as provenance, not as live code.

## New patches (2026-11-22, this program)

### P1 — Local model cards: serve a complete local directory with no hub (F2 fix)
- Files: `src/exo/download/download_utils.py` (+`build_local_model_card`),
  `src/exo/shared/models/model_cards.py` (`ModelCard.load`).
- What was broken: a model that exists only as a local directory (the fleet's
  converted Inkling weights) could never get a ModelCard: `ModelCard.load`
  fell back to `fetch_from_hf`, which requires the model_id to resolve on
  Hugging Face. Without a card there is no placement, no instance, no serve —
  even though the download layer already treats a complete local copy as
  fully downloaded (`store.find_complete`, used by the worker's
  DownloadModel handler and the download coordinator).
- What the patch does: `ModelCard.load`, on a card-cache miss, now tries
  `build_local_model_card` before the HF fetch. The builder returns None
  unless `store.find_complete` finds a complete local copy; it then builds
  the card from the directory itself — ConfigData parsed from its
  config.json (layers, hidden size, KV heads, context length, tensor
  support), storage size = the safetensors bytes on disk, and `model_type`
  from config.json so the master can look up the model's family. The card
  is saved to the custom-cards dir exactly like a fetched one. The HF path
  is untouched and still runs whenever no complete local copy exists.
- Upstream-ability: high — generic (any local-only model benefits), small,
  no behavior change for hub models. Candidate for the upstream PR once the
  stage-4 cluster verification is green.

### P2 — inkling2_loop family: the fleet Inkling-2-Mini architecture (F3 fix)
- Files: `src/exo/worker/engines/mlx/models/inkling2_loop/` (new:
  `model.py`, `sharding.py`, `family.py`, `spec.py`),
  `src/exo/worker/engines/mlx/models/registry.py` (+1 import, +1 FAMILIES
  entry, appended last), `src/exo/worker/engines/mlx/models/spec_registry.py`
  (+1 import, +1 FAMILY_SPECS entry, appended last).
- What was missing: mlx-lm has no class for GGUF architecture
  `inkling2_loop` (24 layers; loop core 6..17 x3 passes with message
  controllers fed from the previous pass's attention-residual block bank;
  hybrid SWA/full attention with relative-position bias, no RoPE; depthwise
  short convs; 128-expert MoE top-8 with SiTU; final-logit softcap).
  Upstream exo ships an `inkling` family, but for a different checkpoint
  (model_type `inkling_mm_model`) whose model.py has no loop, no attn-res
  bank and no message controllers — numerically wrong for this checkpoint,
  so it is not reused.
- What the patch does: a new family whose model.py is a faithful port of
  the fleet llama.cpp graph, verified standalone before this integration
  (stage 3 of the program: greedy agreement with the elven-path
  llama-server to the first divergence, a near-tie argmax flip inside
  representation noise; step-0 logits identical top-6 order). Execution is
  a full-sequence recompute in float32 — the prompt cache is a shared
  token-history object (InklingCache), so exo's prefill/trim/decode flow
  works unchanged. Routed experts (~49 GB of the 52 GB checkpoint) are not
  parameters: sanitize() drops them from the load set and the family's
  post_load re-opens the checkpoint memory-mapped and pages experts in per
  use, so placement math (card memory_size < storage_size) stays honest
  and no rank materializes the other stage's weights.
- Placement: the generic pipeline split cannot express this model (stages
  exchange the residual stream PLUS the block bank, and the loop core must
  stay on one rank), so the family registers a custom `pipeline_parallel`
  hook: exactly 2 ranks, shards (0, 18) and (18, 24) only — anything else
  raises a located error. Rank 0 sends stream+bank; rank 1 runs the
  postlude + output; logits are all_gathered so every rank returns them
  (the deepseek_v4 idiom). Instances are launched with explicit shard
  assignments (POST /instance), since the proportional allocator cannot be
  aimed at the 18/6 split.
- Registration mechanism: the smallest honest one — the family's own
  ModelClasses are picked up by exo's existing mlx-lm registry hook
  (load_model(get_model_classes=...)) from config.json's model_type. No
  loader shim, no mlx-lm patch.
- Upstream-ability: the family is self-contained and additive; the registry
  edits are appends. It serves one specific open checkpoint family; whether
  upstream wants it is their call — the fork carries it either way.

### P3 — warmup length is configurable (EXO_WARMUP_MAX_TOKENS)

- **What:** `warmup_inference` (src/exo/worker/engines/mlx/generator/generate.py) reads the warmup token count from `EXO_WARMUP_MAX_TOKENS`, defaulting to the upstream 50.
- **Why:** warmup's job is to exercise prefill/decode/barrier once before the runner reports ready. For a very slow model on thermally constrained hardware, 50 full-power tokens can exceed the node's thermal envelope before readiness: observed on ni-knight (Ryzen 7 7840HS-class mini-PC) with the inkling2_loop family at ~75 s/token — Tctl hit 91 C sustained during warmup and the run had to be stopped under the project's ~85 C thermal rule. With the variable set to 5, warmup exercises every code path and completes inside the envelope.
- **Upstream-ability:** clean — a one-line environment override with unchanged default behavior; plausible upstream as a deployment knob for edge hardware.


### P4 — runner stall window is configurable (EXO_RUNNER_STALL_TIMEOUT)

- **What:** `RUNNER_STALL_TIMEOUT` (src/exo/worker/runner/supervisor.py) reads `EXO_RUNNER_STALL_TIMEOUT`, defaulting to the upstream 90 s.
- **Why:** the supervisor kills a generating runner that reports no token for the stall window, on the stated assumption that a healthy runner reports a token every step. For a very slow model that assumption inverts: inkling2_loop at ~98 s/token (4 cores, thermally throttled) was killed mid-generation with `model_stalled` ("Runner made no progress for 90s while generating") while computing correctly. Slow-model deployments need a window sized to their step time.
- **Upstream-ability:** clean — an environment override with unchanged default; the failure mode (killing healthy slow runners) is generic to any sufficiently slow model/hardware pairing.


### P5 — MoE experts applied from an affine-quantized resident copy (perf pass 1)

- **What:** `Model._moe` (src/exo/worker/engines/mlx/models/inkling2_loop/model.py)
  gains an expert-major path over affine-quantized experts: on first use of a
  layer, `_ExpertStore.quantized()` builds 8-bit (group_size 64) quantized
  stacks of that layer's gate_up/down expert tensors from the bf16 checkpoint
  and caches them resident; the MoE then applies each distinct expert once per
  call for all (token, slot) pairs routed to it via `mx.quantized_matmul`.
  `EXO_INKLING_EXPERT_BITS=0` restores the legacy path (slot-major, bf16
  experts paged from the memory-mapped checkpoint and widened to f32 per use)
  exactly; the default is 8.
- **Why:** the legacy path re-read and re-widened every routed expert once per
  top-k slot — ~2.2 s per MoE call at T=28 on a Ryzen 7 255 (measured,
  PERF-PASS-1.md), the dominant term in the family's ~0.013 tok/s decode.
  Expert-major alone (identical math, output diff ~1e-7) measured 2.8x; the
  quantized path measured ~114 ms per call (19x) with an MoE-output max diff
  of 8.9e-3 against the bf16 path, inside the representation-noise class the
  stage-3 gate already accepts — and it only pays with the G1b AVX-512
  quantized_matmul kernel in the runner's MLX (stock MLX's scalar CPU
  quantized_matmul eats the gain; PERF-PASS-1.md has the A/B table).
- **Upstream-ability:** the change is contained in the fork-only family; the
  generic lesson (do not re-widen paged experts per slot) is upstreamable
  wherever an MoE pages experts, but the code as written serves this family.

## Findings from the first run (EE 11–21, 2026-11-22) — no code divergence yet

The fleet deployment currently runs VANILLA upstream 0.3.70: clustering,
discovery (zenoh, namespace ranch-exo), Pipeline/Tensor placement and the
ring all work on AMD Ubuntu nodes with zero patches (verified ni-knight +
ashfall-haven; Llama-3.2-1B-Instruct-4bit placed Pipeline 456 MB / 274 MB).
What blocks serving is below exo, in the backend, plus one exo API gap:

### F1 — mlx-cpu (PyPI wheel) is pathologically slow on Ryzen (backend, not exo)
- Raw MLX matmul on ni-knight (Ryzen 7 255): 4096^3 = 6.9 GFLOP/s,
  2304^3 = 7.8 GFLOP/s (hardware should exceed 200). Generate pegs 1 core.
- exo runner warmup (which pre-fills + generates) never completes for the
  4-bit llama graph (two node pairs, 2026-11-21 and 2026-11-22).
- Pure mlx-lm control: 4-bit generate did not finish 60 tokens in 900 s;
  the same model dequantized to bf16 completed at 1.409 t/s.
- Planned fix (half a): build MLX from source on-node (BLAS/AVX-512,
  threading) instead of the PyPI wheel; exo-side, prefer dense/dequantized
  weights on the CPU backend — the modern form of P0's intent. Patch
  entries land here when implemented + measured, not before.

### F2 — exo cannot load local model directories (exo API gap; blocks half b)
- POST /models/add -> ModelCard.fetch_from_hf: model_id must resolve on
  HuggingFace. A local directory path returns 400 "was not found on
  Hugging Face" (verified live; src/exo/api/main.py add_custom_model).
- Fleet weights (converted Inkling MLX dirs) live on fleet storage, not HF.
- Planned patch (half b bridge): local-directory model cards — build the
  card from the directory's config.json + safetensors index and load from
  disk, keeping the HF path untouched.

### F3 — Inkling-2-Mini architecture absent from mlx-lm (half b core)
- GGUF general.architecture = inkling2_loop (custom llama.cpp arch;
  24 layers, loop core layers 6-17 x3 passes, 128-expert MoE top-8,
  no RoPE (relative bias), hybrid SWA/full attention, short convs, SiTU).
- mlx-lm has no such model class. Full spec: the program's STAGE-1.md §5.
- GGUF -> MLX bf16 converter exists (gguf-py dequant, streaming);
  arch module is a named gate (G3) before any Inkling claim is made.

## Upstream PR
Prepared, NOT submitted. Submission gate: exo serving Inkling-2-Mini
sharded across >=2 AMD Ubuntu nodes, measured against the 10.80 t/s
single-node llama-server baseline (Will's first Inkling test, EE 11).
Halves (a) and (b) will be reported separately in the PR body.
