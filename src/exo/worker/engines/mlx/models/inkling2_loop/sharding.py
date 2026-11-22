"""Pipeline placement for inkling2_loop: exactly 2 ranks, split at the loop
boundary (layers 0..17 on rank 0, 18..23 + output on rank 1).

The generic exo split cannot express this model: layers 6..17 execute 3
times as a loop whose later passes consume messages mixed from the previous
pass's attention-residual block bank, so the bank and loop state must stay
on one rank, and the boundary payload is (residual stream + block bank),
not a single hidden tensor. Non-stage parameters are replaced with 0-size
placeholders so the parameter tree keeps every name (exo's load check)
without materializing the other stage's weights.
"""

from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_map_with_path

from exo.shared.types.worker.shards import PipelineShardMetadata

from .model import Model


def pipeline_parallel(
    model: Model,
    group: mx.distributed.Group,
    metadata: PipelineShardMetadata,
) -> Model:
    args = model.args
    if group.size() != 2:
        raise ValueError(
            "inkling2_loop: pipeline placement requires exactly 2 ranks "
            f"(got {group.size()}); the loop core and its message bank are "
            "placed as (0, loop_hi) + (loop_hi, num_layers) and do not "
            "subdivide further."
        )
    start, end = metadata.start_layer, metadata.end_layer
    if (start, end) == (0, args.loop_hi):
        stage = "first"
        keep_prefixes = (
            "model.token_embd",
            "model.token_embd_norm",
        ) + tuple(f"model.layers.{i}." for i in range(0, args.loop_hi))
    elif (start, end) == (args.loop_hi, args.num_hidden_layers):
        stage = "last"
        keep_prefixes = tuple(
            f"model.layers.{i}." for i in range(args.loop_hi, args.num_hidden_layers)
        ) + ("output.", "output_norm.", "output_res_query.")
    else:
        raise ValueError(
            f"inkling2_loop: unsupported shard (start={start}, end={end}); "
            f"supported shards are (0, {args.loop_hi}) and "
            f"({args.loop_hi}, {args.num_hidden_layers}) — the loop core "
            f"(layers {args.loop_lo}..{args.loop_hi - 1}) and its block bank "
            "must stay on one rank."
        )

    def shrink(path: str, value: mx.array) -> mx.array:
        if any(path.startswith(p) for p in keep_prefixes):
            return value
        return mx.zeros((0,), dtype=value.dtype)

    model.update(tree_map_with_path(shrink, model.parameters()))
    model._pp_stage = stage
    model._pp_group = group
    return model
