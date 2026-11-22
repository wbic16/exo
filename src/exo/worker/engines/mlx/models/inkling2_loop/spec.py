from exo.worker.engines.mlx.models.spec import EosRule, FamilySpec

SPEC = FamilySpec(
    name="inkling2_loop",
    model_types=frozenset({"inkling2_loop"}),
    tensor_parallel_architectures=frozenset(),
    eos_rules=(
        # 200006 <|content_model_end_sampling|>: the checkpoint's EOS, which
        # the chat template writes after every assistant turn. Same rule as
        # the upstream inkling family (the model id contains "inkling").
        EosRule(("inkling",), (200006,)),
    ),
)
