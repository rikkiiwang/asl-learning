"""Per-stage size budget (spec §3.7). Caps are int8-quantized; param caps assume
~1 byte/param. Each component plan calls assert_within_budget() before export so
no single model (esp. the landmark model) can eat the whole budget."""
import torch.nn as nn

# (max_params, max_quantized_mb) per component
BUDGETS: dict[str, tuple[float, float]] = {
    "detector":    (2_000_000, 2.0),
    "landmark":    (3_000_000, 6.0),
    "recognizer":  (1_500_000, 3.0),
    "appearance":  (1_000_000, 2.0),
}
HARD_CAP_MB = 20.0      # inviolable total across all components
TARGET_MB = 13.0


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def assert_within_budget(name: str, module: nn.Module) -> None:
    max_params, _ = BUDGETS[name]
    n = count_params(module)
    assert n <= max_params, (
        f"{name} has {n:,} params > cap {int(max_params):,} (spec §3.7); "
        "shrink width/depth rather than borrowing from another stage's budget")


def total_hard_cap_mb() -> float:
    return HARD_CAP_MB
