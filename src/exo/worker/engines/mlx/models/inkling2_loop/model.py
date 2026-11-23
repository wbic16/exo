"""inkling2_loop — the fleet Inkling-2-Mini checkpoint (wbic16/inkling-2-mini-mlx-bf16).

A faithful MLX port of the fleet llama.cpp graph (src/models/inkling2.cpp +
inkling2-loop.cpp), numerically verified in stage 3 of the Inkling-on-exo
program: greedy agreement with the elven-path llama-server on the gate prompt
to the first divergence, itself a near-tie argmax flip inside representation
noise (see ~/workspace/exo-integration/INKLING-ON-EXO.md, stage 3).

Architecture notes (why this family cannot use exo's generic pipeline split):
- Layers 6..17 form a loop core executed 3 times; later passes are fed
  "messages" mixed from the previous pass's attention-residual block bank.
  The bank and the loop state must live on one rank, so pipeline placement is
  only supported as (0, 18) + (18, 24) across exactly 2 ranks — see
  sharding.py. Rank 0 runs embed + prelude + loop and sends the residual
  stream plus the 3-bank blocks; rank 1 runs the postlude + output stage;
  the logits are all_gathered so every rank returns them (the deepseek_v4
  idiom).
- Execution is a full-sequence recompute: there is no incremental KV state.
  The prompt cache is therefore a shared token-history object (InklingCache);
  each forward recomputes the whole history in float32, exactly like the
  verified stage-3 module. Correct, and slow — speed is the G1/G1b kernel
  program's problem, not this family's.
- Routed experts (48 tensors, ~49 GB of the 52 GB checkpoint) are NOT model
  parameters: they stay memory-mapped on disk and are paged in per use by
  the MoE, loaded by the family's post_load hook into an opaque store.
  Since P5 the MoE applies them from an affine-quantized resident copy
  built lazily per layer (EXO_INKLING_EXPERT_BITS, default 8; 0 restores
  the paged bf16 path).
  sanitize() drops them from the load set so exo's missing-parameter check
  and its mx.eval(model.parameters()) never try to materialize them.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache

_EXPERT_SUFFIXES = ("ffn_gate_up_exps.weight", "ffn_down_exps.weight")

# P5: the MoE applies routed experts from an affine-quantized resident copy
# (see _ExpertStore.quantized), built lazily per layer on first use.
# EXO_INKLING_EXPERT_BITS=0 restores the legacy bf16 page-in path exactly.
_EXPERT_BITS = int(os.environ.get("EXO_INKLING_EXPERT_BITS", "8"))
_EXPERT_GROUP_SIZE = 64


@dataclass
class ModelArgs:
    model_type: str = "inkling2_loop"
    hidden_size: int = 2304
    num_hidden_layers: int = 24
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    head_dim: int = 128
    vocab_size: int = 200058
    padded_vocab_size: int = 201024
    rms_norm_eps: float = 1e-6
    loop_lo: int = 6
    loop_hi: int = 18
    loop_passes: int = 3
    gate_rank: int = 64
    attn_res_block: int = 6
    full_layers: tuple[int, ...] = (5, 11, 17, 23)
    sliding_window: int = 1024
    rel_length: int = 16
    rel_extent: int = 1024
    rel_clip: float = 8.0
    qk_clip: float = 22.627416610717773
    log_scale_alpha: float = 0.1
    log_scale_floor: float = 8192.0
    situ_beta: float = 4.0
    situ_lb: float = 25.0
    softcap: float = 30.0
    expert_scale: float = 8.0
    num_experts: int = 128
    top_k: int = 8
    expert_ffn_length: int = 1152
    conv_cache: int = 4

    @classmethod
    def from_dict(cls, params: dict[str, Any]) -> "ModelArgs":
        known = set(cls.__dataclass_fields__)
        args = cls(**{k: v for k, v in params.items() if k in known})
        args.full_layers = tuple(args.full_layers)
        return args


# ---------------------------------------------------------------------------
# primitives, transcribed from the verified stage-3 module (ggml semantics)
# ---------------------------------------------------------------------------


def _rms(x: mx.array, weight: mx.array | None = None, eps: float = 1e-6) -> mx.array:
    """ggml_rms_norm (+ optional mul weight): x / sqrt(mean(x^2)+eps)."""
    ms = mx.mean(x * x, axis=-1, keepdims=True)
    y = x * mx.rsqrt(ms + eps)
    return y if weight is None else y * weight


def _clip_rows(x: mx.array, scale: mx.array | None, thr: float) -> mx.array:
    """graph::clip — optional per-channel scale, then scale each row down to
    L2 norm <= thr: factor = clamp(thr / (||x|| + 1e-6), 0, 1)."""
    if scale is not None:
        x = x * scale
    n = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True)) + 1e-6
    f = mx.minimum(thr / n, 1.0)
    return x * f


def _mix(blocks: list[mx.array], cur: mx.array, query: mx.array, emb: int) -> mx.array:
    """graph::mix — depth attention over the block bank + current stream.
    blocks: list of [T, E]; cur: [T, E]; query: [C, E]. Returns [T, C, E]."""
    s = mx.stack(blocks + [cur], axis=1)  # [T, D, E]
    s = _rms(s)
    sc = mx.matmul(s, query.T) * (1.0 / math.sqrt(emb))  # [T, D, C]
    wts = mx.softmax(sc, axis=1)
    return mx.matmul(wts.transpose(0, 2, 1), s)  # [T, C, E]


def _convolve(
    x: mx.array, wgt: mx.array, dyn: mx.array | None, nconv: int
) -> mx.array:
    """graph::convolve — causal depthwise conv (kernel 4, zero history) with
    residual, scaled by 1/sqrt((dyn?2:1)*4 + 1). x: [T, C]; wgt: [C, 4];
    dyn: [T, H, 4] input-dependent per-head taps or None."""
    t, c = x.shape
    xpad = mx.concatenate([mx.zeros((nconv - 1, c), dtype=x.dtype), x], axis=0)
    out = mx.zeros_like(x)
    for lag in range(nconv):
        out = out + xpad[lag : lag + t] * wgt[:, lag]
    if dyn is not None:
        h = dyn.shape[1]
        ch = c // h
        xr = xpad.reshape(t + nconv - 1, h, ch)
        for lag in range(nconv):
            win = xr[nconv - 1 - lag : nconv - 1 - lag + t]  # [T, H, ch]
            out = out + (win * dyn[:, :, lag][:, :, None]).reshape(t, c)
    scale = 1.0 / math.sqrt((2 if dyn is not None else 1) * nconv + 1)
    return (x + out) * scale


def _situ(gate: mx.array, up: mx.array, beta: float, lb: float) -> mx.array:
    """situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb)."""
    act = beta * mx.tanh(gate / beta) * mx.sigmoid(gate)
    return act * (lb * mx.tanh(up / lb))


# ---------------------------------------------------------------------------
# cache: the shared token history IS the state (full-sequence recompute)
# ---------------------------------------------------------------------------


class _History:
    __slots__ = ("tokens",)

    def __init__(self) -> None:
        self.tokens: list[int] = []


class InklingCache(KVCache):
    """One entry per layer, all sharing a single _History.

    keys/values stay None — the model recomputes attention from the token
    history on every call. `offset` mirrors len(history) so mlx_lm/exo
    bookkeeping (trim, prefix-cache lengths) behaves; trim() truncates the
    shared history once (the first entry trimmed does it; the rest no-op).
    """

    def __init__(self, history: _History | None = None) -> None:
        super().__init__()
        self.history = history if history is not None else _History()

    @property
    def state(self) -> tuple[None, None]:  # type: ignore[override]
        return (None, None)

    @state.setter
    def state(self, v: Any) -> None:
        if v and v[0] is not None:
            self.offset = int(v[0].shape[-2])

    def trim(self, n: int) -> int:  # type: ignore[override]
        trimmed = super().trim(n)
        h = self.history.tokens
        if len(h) > self.offset:
            del h[self.offset :]
        return trimmed


# ---------------------------------------------------------------------------
# parameter holders (names must equal the GGUF tensor names after sanitize)
# ---------------------------------------------------------------------------


class _W(nn.Module):
    def __init__(self, *shape: int) -> None:
        super().__init__()
        self.weight = mx.zeros(shape)


class _B(nn.Module):
    def __init__(self, *shape: int) -> None:
        super().__init__()
        self.bias = mx.zeros(shape)


class _ExpertStore:
    """Opaque (non-array) leaf: routed experts paged from disk by the MoE.

    Deliberately NOT an nn.Module / dict / list, so mlx's parameter tree
    never sees the expert tensors."""

    def __init__(self) -> None:
        self.by_layer: dict[int, tuple[mx.array, mx.array]] = {}
        self._quantized: dict[int, tuple[mx.array, ...]] = {}

    def quantized(
        self, il: int, gu_all: mx.array, dn_all: mx.array
    ) -> tuple[mx.array, ...]:
        """Affine-quantized expert stacks for layer il (P5), built once on
        first use and cached. Returns (gu_q, gu_scales, gu_biases, dn_q,
        dn_scales, dn_biases), each stacked over the expert axis. Only the
        layers a rank actually runs are ever built."""
        got = self._quantized.get(il)
        if got is None:
            parts: list[list[mx.array]] = [[], [], [], [], [], []]
            for e in range(gu_all.shape[0]):
                wq, s, b = mx.quantize(
                    gu_all[e].astype(mx.float32),
                    group_size=_EXPERT_GROUP_SIZE,
                    bits=_EXPERT_BITS,
                )
                parts[0].append(wq)
                parts[1].append(s)
                parts[2].append(b)
                wq, s, b = mx.quantize(
                    dn_all[e].astype(mx.float32),
                    group_size=_EXPERT_GROUP_SIZE,
                    bits=_EXPERT_BITS,
                )
                parts[3].append(wq)
                parts[4].append(s)
                parts[5].append(b)
            got = tuple(mx.stack(p) for p in parts)
            mx.eval(got)
            self._quantized[il] = got
        return got


class InklingLayer(nn.Module):
    def __init__(self, args: ModelArgs, il: int) -> None:
        super().__init__()
        e = args.hidden_size
        nh, nkv, hd = args.num_attention_heads, args.num_key_value_heads, args.head_dim
        nrel, ext = args.rel_length, args.rel_extent
        nconv = args.conv_cache
        full = il in args.full_layers
        loop = args.loop_lo <= il < args.loop_hi

        self.attn_q = _W(nh * hd, e)
        self.attn_k = _W(nkv * hd, e)
        self.attn_v = _W(nkv * hd, e)
        self.attn_gate = _W(nh * hd, e)
        self.attn_output = _W(e, nh * hd)
        self.attn_sink_proj = _W(nh, e)
        self.attn_rel = _W(nh * nrel, e)
        self.attn_rel_bias = _W(ext, nrel)
        self.attn_q_clip = _W(nh, hd)
        self.attn_rel_clip = _W(nh, nrel)
        self.attn_k_conv = _W(nkv * hd, nconv)
        self.attn_v_conv = _W(nkv * hd, nconv)
        self.attn_post_conv = _W(e, nconv)
        self.ffn_post_conv = _W(e, nconv)
        self.ffn_gate_inp = _W(args.num_experts, e)
        self.exp_probs_b = _B(args.num_experts)
        self.ffn_res_query = _W(e)
        self.ffn_norm = _W(e)
        if il > 0:
            self.attn_res_query = _W(3, e) if full else _W(e)
        if full:
            self.attn_q_inp_norm = _W(e)
            self.attn_k_inp_norm = _W(e)
            self.attn_v_inp_norm = _W(e)
            self.attn_k_conv_proj = _W(nkv * nconv, e)
            self.attn_v_conv_proj = _W(nkv * nconv, e)
        else:
            self.attn_norm = _W(e)
        if loop:
            self.loop_msg_query = _W(4, e)
            for name in ("q", "k", "v", "ffn"):
                setattr(self, f"loop_{name}_ctrl_down", _W(args.gate_rank, 2 * e))
                setattr(self, f"loop_{name}_ctrl_gain", _W(e, args.gate_rank))
                setattr(self, f"loop_{name}_ctrl_forget", _W(e, args.gate_rank))


class _Inner(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.token_embd = _W(args.padded_vocab_size, args.hidden_size)
        self.token_embd_norm = _W(args.hidden_size)
        self.layers = [InklingLayer(args, il) for il in range(args.num_hidden_layers)]


def _f32(a: mx.array) -> mx.array:
    return a.astype(mx.float32)


class Model(nn.Module):
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        self.args = args
        self.model = _Inner(args)
        self.output = _W(args.padded_vocab_size, args.hidden_size)
        self.output_norm = _W(args.hidden_size)
        self.output_res_query = _W(args.hidden_size)
        self._experts = _ExpertStore()
        # pipeline state, set by sharding.pipeline_parallel
        self._pp_stage = "full"
        self._pp_group: Any = None

    # -- loading -----------------------------------------------------------

    @staticmethod
    def sanitize(weights: dict[str, mx.array]) -> dict[str, mx.array]:
        out: dict[str, mx.array] = {}
        for k, v in weights.items():
            if k.endswith(_EXPERT_SUFFIXES):
                continue  # paged from disk by post_load, never parameters
            if k == "token_embd.weight":
                out["model.token_embd.weight"] = v
            elif k == "token_embd_norm.weight":
                out["model.token_embd_norm.weight"] = v
            elif k.startswith("blk."):
                out["model.layers." + k[4:]] = v
            elif k in ("output.weight", "output_norm.weight", "output_res_query.weight"):
                out[k] = v
        return out

    @property
    def layers(self) -> list[InklingLayer]:
        # exo's cache factory asserts model.layers before it will use
        # make_cache (the mlx-lm convention: a property onto the inner model).
        return self.model.layers

    def make_cache(self) -> list[InklingCache]:
        history = _History()
        return [InklingCache(history) for _ in range(self.args.num_hidden_layers)]

    # -- pieces ------------------------------------------------------------

    def _control(
        self, layer: InklingLayer, cur: mx.array, msg: mx.array, name: str
    ) -> mx.array:
        """graph::control — (1 - forget) * cur + gain * msg, gates tanh of a
        rank-64 bottleneck over concat(rms(cur), rms(msg))."""
        g = mx.concatenate([_rms(cur), _rms(msg)], axis=-1)  # [T, 2E]
        h = g @ _f32(getattr(layer, f"loop_{name}_ctrl_down").weight).T
        gain = mx.tanh(h @ _f32(getattr(layer, f"loop_{name}_ctrl_gain").weight).T)
        forget = mx.tanh(h @ _f32(getattr(layer, f"loop_{name}_ctrl_forget").weight).T)
        return cur * (1.0 - forget) + gain * msg

    def _moe(self, layer: InklingLayer, il: int, x: mx.array) -> mx.array:
        """build_moe_ffn: sigmoid router (F32), selection probs + exp_probs_b,
        top-8; weights = unbiased probs, normalized by (sum + 1e-20), x scale.
        gate = first half of gate_up rows, up = second half."""
        args = self.args
        logits = x @ _f32(layer.ffn_gate_inp.weight).T  # [T, 128]
        probs = mx.sigmoid(logits)
        sel = probs + _f32(layer.exp_probs_b.bias)
        idx = mx.argpartition(-sel, kth=args.top_k - 1, axis=-1)[:, : args.top_k]
        wsel = mx.take_along_axis(probs, idx, axis=-1)
        wsel = wsel / (mx.sum(wsel, axis=-1, keepdims=True) + 1e-20) * args.expert_scale
        mx.eval(idx, wsel)
        idx_np = idx.tolist()
        out = mx.zeros_like(x)
        gu_all, dn_all = self._experts.by_layer[il]  # bf16, memory-mapped
        if _EXPERT_BITS > 0:
            # P5: expert-major application over the quantized resident copy.
            # Each distinct expert is matmul'd once per call for every
            # (token, slot) pair that routed to it, instead of being paged
            # in and widened to f32 once per slot (the legacy path below).
            pairs: dict[int, list[tuple[int, int]]] = {}
            for t, row in enumerate(idx_np):
                for slot, e in enumerate(row):
                    pairs.setdefault(e, []).append((t, slot))
            guq, gus, gub, dnq, dns, dnb = self._experts.quantized(il, gu_all, dn_all)
            for e, ts in pairs.items():
                toks = [t for t, _ in ts]
                slots = [s for _, s in ts]
                xt = x[toks]  # [n, E]
                h = mx.quantized_matmul(
                    xt, guq[e], gus[e], gub[e],
                    transpose=True, group_size=_EXPERT_GROUP_SIZE, bits=_EXPERT_BITS,
                )
                act = _situ(h[:, : args.expert_ffn_length], h[:, args.expert_ffn_length :], args.situ_beta, args.situ_lb)
                o = mx.quantized_matmul(
                    act, dnq[e], dns[e], dnb[e],
                    transpose=True, group_size=_EXPERT_GROUP_SIZE, bits=_EXPERT_BITS,
                )
                o = o * wsel[toks, slots][:, None]
                out = out.at[toks].add(o)
            return out
        for slot in range(args.top_k):
            by_exp: dict[int, list[int]] = {}
            for t, e in enumerate(row[slot] for row in idx_np):
                by_exp.setdefault(e, []).append(t)
            for e, toks in by_exp.items():
                xt = x[toks]  # [n, E]
                gu = gu_all[e].astype(mx.float32)  # [2304, 2304]
                h = xt @ gu.T
                act = _situ(h[:, : args.expert_ffn_length], h[:, args.expert_ffn_length :], args.situ_beta, args.situ_lb)
                dn = dn_all[e].astype(mx.float32)  # [2304, 1152]
                o = (act @ dn.T) * wsel[toks, slot][:, None]
                out = out.at[toks].add(o)
        return out

    def _attention(
        self,
        q: mx.array,
        k: mx.array,
        v: mx.array,
        r: mx.array,
        bias_w: mx.array,
        sink: mx.array,
        is_swa: bool,
    ) -> mx.array:
        """Q [T, NH, HD] (clipped, tau-scaled), K/V [T, NKV, HD],
        R [T, NH, NREL], bias_w [EXT, NREL], sink [T, NH]."""
        args = self.args
        nh, nkv, hd = args.num_attention_heads, args.num_key_value_heads, args.head_dim
        ext = args.rel_extent
        t = q.shape[0]
        ke = mx.repeat(k, nh // nkv, axis=1)  # [T, NH, HD]
        ve = mx.repeat(v, nh // nkv, axis=1)
        sc = mx.matmul(q.transpose(1, 0, 2), ke.transpose(1, 2, 0)) * (1.0 / math.sqrt(hd))
        # rel bias: table[h, t, e] = sum_r bias_w[e, r] * R[t, h, r]; gather by distance
        tab = mx.matmul(r.transpose(1, 0, 2), bias_w.T)  # [NH, T, EXT]
        tab = tab - sink.transpose(1, 0)[:, :, None]  # fold -t
        pos = mx.arange(t)
        d = pos[:, None] - pos[None, :]  # [Tq, Tk]
        didx = mx.where((d >= 0) & (d < ext), d, mx.full(d.shape, ext, dtype=d.dtype))
        bias = mx.take_along_axis(tab, didx[None, :, :], axis=2)  # [NH, Tq, Tk]
        sc = sc + bias
        neg = mx.full(sc.shape, -1e30, dtype=sc.dtype)
        bad = d < 0
        if is_swa:
            bad = bad | (d >= args.sliding_window)
        sc = mx.where(bad[None, :, :], neg, sc)
        # sink: append a zero-logit column, softmax, drop it
        sc = mx.concatenate([sc, mx.zeros((nh, t, 1), dtype=sc.dtype)], axis=-1)
        p = mx.softmax(sc, axis=-1)[:, :, :-1]
        out = mx.matmul(p, ve.transpose(1, 0, 2))  # [NH, T, HD]
        return out.transpose(1, 0, 2).reshape(t, nh * hd)

    def _tau_vec(self, t: int) -> mx.array:
        """log-scaled position factor for full-attention layers:
        1 + alpha * log(max(1, (pos + 1) / floor))."""
        args = self.args
        pos = mx.arange(t).astype(mx.float32)
        tt = mx.maximum((pos + 1.0) / args.log_scale_floor, 1.0)
        return 1.0 + args.log_scale_alpha * mx.log(tt)

    def _layer(
        self,
        il: int,
        ip: int,
        cur: mx.array,
        blocks: list[mx.array],
        messages: list[mx.array] | None,
    ) -> mx.array:
        """One physical layer; `blocks` is mutated in place (the attn-res
        block bank), exactly as the C++ vector. messages: 4 [T, E] tensors
        (q, k, v, ffn) or None."""
        args = self.args
        layer = self.model.layers[il]
        t = cur.shape[0]
        is_swa = il not in args.full_layers
        inp_sa = cur

        # attention inputs via depth-mix (layer 0 has no attn_res_query)
        if il > 0:
            q = _f32(layer.attn_res_query.weight)
            if q.ndim == 1:
                q = q[None, :]
            mixed = _mix(blocks, cur, q, args.hidden_size)  # [T, C, E]
            if is_swa:
                inputs = [mixed[:, 0], mixed[:, 0], mixed[:, 0]]
            else:
                inputs = [mixed[:, 0], mixed[:, 1], mixed[:, 2]]
        else:
            inputs = [cur, cur, cur]

        if messages is not None:
            names = ["q", "k", "v"]
            inputs = [self._control(layer, inputs[j], messages[j], names[j]) for j in range(3)]

        if is_swa:
            nw = _f32(layer.attn_norm.weight)
            if messages is not None:
                inputs = [_rms(inp, nw, args.rms_norm_eps) for inp in inputs]
            else:
                n0 = _rms(inputs[0], nw, args.rms_norm_eps)
                inputs = [n0, n0, n0]
        else:
            inputs = [
                _rms(inputs[0], _f32(layer.attn_q_inp_norm.weight), args.rms_norm_eps),
                _rms(inputs[1], _f32(layer.attn_k_inp_norm.weight), args.rms_norm_eps),
                _rms(inputs[2], _f32(layer.attn_v_inp_norm.weight), args.rms_norm_eps),
            ]
        q_in, k_in, v_in = inputs

        # self-attention
        nh, nkv, hd = args.num_attention_heads, args.num_key_value_heads, args.head_dim
        q = (q_in @ _f32(layer.attn_q.weight).T).reshape(t, nh, hd)
        kraw = k_in @ _f32(layer.attn_k.weight).T  # [T, 1024]
        dyn_k = None
        if not is_swa:
            dyn_k = (k_in @ _f32(layer.attn_k_conv_proj.weight).T).reshape(t, nkv, args.conv_cache)
        k = _convolve(kraw, _f32(layer.attn_k_conv.weight), dyn_k, args.conv_cache).reshape(t, nkv, hd)
        vraw = v_in @ _f32(layer.attn_v.weight).T
        dyn_v = None
        if not is_swa:
            dyn_v = (v_in @ _f32(layer.attn_v_conv_proj.weight).T).reshape(t, nkv, args.conv_cache)
        v = _convolve(vraw, _f32(layer.attn_v_conv.weight), dyn_v, args.conv_cache).reshape(t, nkv, hd)
        r = (q_in @ _f32(layer.attn_rel.weight).T).reshape(t, nh, args.rel_length)

        q = _clip_rows(q, _f32(layer.attn_q_clip.weight), args.qk_clip)
        k = _clip_rows(k, None, args.qk_clip)
        v = _rms(v)  # plain rms_norm over head dim
        r = _clip_rows(r, _f32(layer.attn_rel_clip.weight), args.rel_clip)

        if not is_swa:
            tv = self._tau_vec(t)[:, None, None]
            q = q * tv
            r = r * tv

        sink = q_in @ _f32(layer.attn_sink_proj.weight).T  # [T, NH]
        att = self._attention(q, k, v, r, _f32(layer.attn_rel_bias.weight), sink, is_swa)
        att = att * mx.sigmoid(q_in @ _f32(layer.attn_gate.weight).T)
        cur = _convolve(att @ _f32(layer.attn_output.weight).T, _f32(layer.attn_post_conv.weight), None, args.conv_cache)

        # block attention-residual bookkeeping
        if ip % args.attn_res_block == 0:
            blocks.append(inp_sa)
        else:
            cur = inp_sa + cur

        # MoE FFN
        ffn_inp = cur
        fq = _f32(layer.ffn_res_query.weight)
        f = _mix(blocks, cur, fq[None, :], args.hidden_size)[:, 0]
        if messages is not None:
            f = self._control(layer, f, messages[3], "ffn")
        f = _rms(f, _f32(layer.ffn_norm.weight), args.rms_norm_eps)
        f = self._moe(layer, il, f)
        f = _convolve(f, _f32(layer.ffn_post_conv.weight), None, args.conv_cache)
        return ffn_inp + f

    def _embed_prelude_loop(self, ids: mx.array) -> tuple[mx.array, list[mx.array]]:
        args = self.args
        cur = self.model.token_embd.weight[ids].astype(mx.float32)
        cur = _rms(cur, _f32(self.model.token_embd_norm.weight), args.rms_norm_eps)
        blocks: list[mx.array] = []
        for il in range(args.loop_lo):
            cur = self._layer(il, il, cur, blocks, None)
        prelude = cur
        prelude_blocks = list(blocks)
        for p in range(args.loop_passes):
            if p == 0:
                for ip in range(args.loop_lo, args.loop_hi):
                    cur = self._layer(ip, ip, cur, blocks, None)
            else:
                bank = blocks[len(prelude_blocks) + 1 :]
                bank_out = cur
                cur = prelude
                blocks = list(prelude_blocks)
                for ip in range(args.loop_lo, args.loop_hi):
                    layer = self.model.layers[ip]
                    mm = _mix(bank, bank_out, _f32(layer.loop_msg_query.weight), args.hidden_size)
                    messages = [mm[:, c] for c in range(4)]
                    cur = self._layer(ip, ip, cur, blocks, messages)
        return cur, blocks

    def _postlude_output(self, cur: mx.array, blocks: list[mx.array]) -> mx.array:
        args = self.args
        for il in range(args.loop_hi, args.num_hidden_layers):
            cur = self._layer(il, il, cur, blocks, None)
        mixed = _mix(blocks, cur, _f32(self.output_res_query.weight)[None, :], args.hidden_size)[:, 0]
        h = _rms(mixed, _f32(self.output_norm.weight), args.rms_norm_eps)
        logits = h @ _f32(self.output.weight).T  # [T, padded_vocab]
        logits = args.softcap * mx.tanh(logits / args.softcap)
        return logits[:, : args.vocab_size]

    # -- forward -----------------------------------------------------------

    def __call__(
        self,
        inputs: mx.array,
        cache: list[InklingCache] | None = None,
        input_embeddings: mx.array | None = None,
    ) -> mx.array:
        args = self.args
        ids = inputs[0]  # [T_new], batch 1 (SequentialGenerator)
        if cache is not None:
            history = cache[0].history
            history.tokens.extend(int(t) for t in ids.tolist())
            tokens = history.tokens
        else:
            tokens = [int(t) for t in ids.tolist()]
        t = len(tokens)
        stage = self._pp_stage

        result: mx.array
        if stage == "last":
            group = self._pp_group
            n_bank = len([ip for ip in range(args.loop_hi) if ip % args.attn_res_block == 0])
            payload = mx.distributed.recv_like(
                mx.zeros((1 + n_bank, t, args.hidden_size), dtype=mx.float32),
                0,
                group=group,
            )
            cur = payload[0]
            blocks = [payload[i] for i in range(1, 1 + n_bank)]
            logits = self._postlude_output(cur, blocks)
            gathered = mx.distributed.all_gather(logits[None, :, :], group=group)
            # Materialize before returning: a lazy collective left pending
            # across the call boundary can pair with the wrong collective on
            # the other rank (exo's pre-decode mx_barrier deadlocked on
            # exactly this). PipelineLastLayer evals before send for the
            # same reason.
            mx.eval(gathered)
            result = gathered[group.size() - 1 : group.size()]
        else:
            cur, blocks = self._embed_prelude_loop(mx.array(tokens, dtype=mx.int32))
            if stage == "first":
                group = self._pp_group
                payload = mx.concatenate(
                    [cur[None, :, :], mx.stack(blocks, axis=0)], axis=0
                )  # [1 + n_bank, T, E]
                mx.eval(payload)
                sent = mx.distributed.send(payload, 1, group=group)
                # Eval the send itself: an unevaluated send may never be
                # driven by the lazy graph, leaving the peer's recv (and
                # then every later collective) waiting forever — the
                # two-process repro deadlocked on exactly this.
                mx.eval(sent)
                zeros = mx.zeros((1, t, args.vocab_size), dtype=mx.float32)
                gathered = mx.distributed.all_gather(zeros, group=group)
                mx.eval(gathered)  # see the note on the "last" branch
                result = gathered[group.size() - 1 : group.size()]
            else:
                result = self._postlude_output(cur, blocks)[None, :, :]

        if cache is not None:
            for c in cache:
                c.offset = t
        return result
