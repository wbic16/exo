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

(none yet — entries land here as the run produces them, one per file, with
what was broken, what the patch does, and the hardware evidence.)

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
