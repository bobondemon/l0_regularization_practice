import hydra
import torch
from omegaconf import DictConfig
import pytorch_lightning as pl
from pathlib import Path


@hydra.main(config_path="conf", config_name="test_config")
def test(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)
    module = hydra.utils.instantiate(cfg.module)
    trainer = hydra.utils.instantiate(cfg.trainer)

    if cfg.ckpt_path is not None:
        ckpt_path = Path.cwd() / cfg.ckpt_path
        assert ckpt_path.is_file(), f"[Error]: no such file {ckpt_path}"
        print(f"[Info]: Load from ckpt path = {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        module.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        print("[Error]: ckpt_path is None")

    test_dataloader = hydra.utils.instantiate(cfg.test_dataloader)

    # Test
    print("[Info]: Start testing")
    test_result = trainer.test(module, dataloaders=test_dataloader, verbose=False)
    test_acc = test_result[0]["test_acc"]
    print(f"test_acc = {test_acc}")


if __name__ == "__main__":
    test()
