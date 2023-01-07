import logging
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pathlib import Path


@hydra.main(version_base=None, config_path="conf", config_name="train_config")
def train(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)
    module = hydra.utils.instantiate(cfg.module)
    cfg_trainer = cfg.trainer

    if cfg.ckpt_path is not None:
        print(f"Path.cwd()={Path.cwd()}")
        ckpt_path = Path.cwd() / cfg.ckpt_path
        logging.info(f"[Info]: Load from ckpt path = {ckpt_path}")
        print(f"[Info]: Load from ckpt path = {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        module.load_state_dict(ckpt["state_dict"], strict=False)
        if cfg.resume_training and not cfg.without_using_gate:
            assert 1 == 2, "[Error]: resume_training should be false if you let without_using_gate=false"
        if cfg.resume_training and cfg.without_using_gate:
            print("[Info]: Resume Training ...")
            OmegaConf.set_struct(cfg_trainer, False)
            cfg_trainer["resume_from_checkpoint"] = cfg.ckpt_path
            OmegaConf.set_struct(cfg_trainer, True)
    else:
        print("[Info]: Training from scratch ...")

    trainer = hydra.utils.instantiate(cfg_trainer)

    train_dataloader = hydra.utils.instantiate(cfg.train_dataloader)
    val_dataloader = hydra.utils.instantiate(cfg.val_dataloader)
    test_dataloader = hydra.utils.instantiate(cfg.test_dataloader)

    # Train
    print("[Info]: Starting training!")
    logging.info("[Info]: Starting training!")
    trainer.fit(module, train_dataloader, val_dataloader)
    # Test best model on validation and test set
    val_result = trainer.test(module, dataloaders=val_dataloader, verbose=False)
    test_result = trainer.test(module, dataloaders=test_dataloader, verbose=False)
    result = {"test": test_result[0]["test_acc"], "val": val_result[0]["test_acc"]}
    print(result)


if __name__ == "__main__":
    train()
