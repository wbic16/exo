"""Family wiring for inkling2_loop: model classes, the custom pipeline
hook, and the post_load hook that pages the routed experts in from disk
(memory-mapped) and pre-casts the dense weights to float32.

The pre-cast is numerically neutral (the stage-3 reference casts every
dense weight to f32 at each use; doing it once at load only removes
repeated cast traffic — measured on ni-knight: a T=7 forward drops from
~42 s to ~16 s). Experts stay bf16 and are cast per use inside the MoE,
exactly as the reference does.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_map

from exo.worker.engines.mlx.models.family import (
    Architecture,
    ModelClasses,
    ModelFamily,
)

from .model import Model, ModelArgs
from .sharding import pipeline_parallel
from .spec import SPEC


def _post_load(model: Model, model_path: Path) -> None:
    # Routed experts: re-open the checkpoint memory-mapped and stash the two
    # expert tensors per layer in the model's opaque store. They are not
    # parameters (sanitize dropped them), so nothing here is materialized.
    weights = mx.load(str(model_path / "model.safetensors"))
    for il in range(model.args.num_hidden_layers):
        model._experts.by_layer[il] = (
            weights[f"blk.{il}.ffn_gate_up_exps.weight"],
            weights[f"blk.{il}.ffn_down_exps.weight"],
        )

    # Pre-cast dense parameters to f32 once (see module docstring). 0-size
    # pipeline placeholders are skipped.
    def cast(value: mx.array) -> mx.array:
        if value.size == 0 or value.dtype == mx.float32:
            return value
        return value.astype(mx.float32)

    model.update(tree_map(cast, model.parameters()))


FAMILY = ModelFamily(
    spec=SPEC,
    architectures=(
        Architecture(
            model_type="inkling2_loop",
            model_class=Model,
            pipeline_parallel=pipeline_parallel,
            batched_generation=False,
            post_load=_post_load,
        ),
    ),
    model_classes=(
        ModelClasses(
            model_type="inkling2_loop",
            model_class=Model,
            model_args_class=ModelArgs,
        ),
    ),
)
