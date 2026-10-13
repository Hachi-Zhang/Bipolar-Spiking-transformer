"""Fine-tune the model from checkpoint-191.pth for a short second-stage run.

This entry point keeps the model definition and data pipeline in
``main_finetune.py`` while providing safe defaults for the user's checkpoint.
Command-line arguments can still override every default.
"""

import os
from pathlib import Path

from main_finetune import get_args_parser, main


def build_args():
    parser = get_args_parser()
    parser.set_defaults(
        finetune="./checkpoint-191.pth",
        model="Efficient_Spiking_Transformer_l",
        batch_size=32,
        num_workers=4,
        epochs=30,
        warmup_epochs=3,
        # A second-stage fine-tune should make small updates to the source model.
        blr=1e-5,
        weight_decay=0.05,
        output_dir="./outputs/finetune-191",
        log_dir="./outputs/finetune-191",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # This entry point is intentionally single-GPU. Cluster launchers can
    # leave distributed variables in the environment even for plain python.
    for env_name in (
        "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE",
        "GROUP_RANK", "ROLE_RANK",
    ):
        os.environ.pop(env_name, None)
    args = build_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
