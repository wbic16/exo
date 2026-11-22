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
