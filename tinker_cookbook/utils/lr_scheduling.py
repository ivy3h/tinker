import logging
import math

logger = logging.getLogger(__name__)


def compute_schedule_lr_multiplier(
    lr_schedule: str, step: int, total_steps: int, warmup_ratio: float = 0.05
) -> float:
    """
    What factor to multiply the base LR by due to the LR schedule
    """
    warmup_steps = int(total_steps * warmup_ratio)

    # Warmup phase: linear increase from 0 to 1
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))

    # Main schedule
    if lr_schedule == "linear":
        # Linear decay from 1.0 to 0.0 after warmup
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    elif lr_schedule == "constant":
        return 1.0

    elif lr_schedule == "cosine":
        # Cosine annealing from 1.0 to 0.0 after warmup
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    else:
        raise ValueError(f"Unknown learning rate schedule: {lr_schedule}")
