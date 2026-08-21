from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf

from pretrain import PretrainConfig, create_dataloader, create_model


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"

HIDDEN_SIZES = [64, 96, 128]


def build_config(hidden_size: int) -> PretrainConfig:
    with hydra.initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIR),
    ):
        cfg = hydra.compose(
            config_name="cfg_baseline_v2",
            overrides=[
                "arch.halt_max_steps=4",
                f"arch.hidden_size={hidden_size}",
                "device=cpu",
                "compile_model=false",
                "dataloader_num_workers=0",
            ],
        )

    return PretrainConfig(
        **OmegaConf.to_container(cfg, resolve=True)
    )


def main():
    device = torch.device("cpu")

    print("=" * 72)
    print("TRM PARAMETER COUNT â€” 4-STEP BASELINE")
    print("=" * 72)

    for hidden_size in HIDDEN_SIZES:
        config = build_config(hidden_size)

        _, metadata = create_dataloader(
            config,
            "train",
            rank=0,
            world_size=1,
            device=device,
            test_set_mode=False,
            epochs_per_iter=1,
            global_batch_size=config.global_batch_size,
        )

        model, _, _ = create_model(
            config,
            metadata,
            rank=0,
            world_size=1,
            device=device,
        )

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        print()
        print(f"hidden_size = {hidden_size}")
        print(f"total parameters     = {total:,}")
        print(f"trainable parameters = {trainable:,}")

        del model


if __name__ == "__main__":
    main()
