import hydra
from omegaconf import DictConfig
import time


@hydra.main(config_path="conf", config_name="sim_datamodule")
def sim(cfg: DictConfig) -> None:
    """
    An example usage of traversing datamodule
    """
    dataloader = hydra.utils.instantiate(cfg.dataloader)
    print("Start traversing dataset")
    start_sec = time.time()
    for _ in dataloader:
        pass
    dur_sec = time.time() - start_sec
    print(f"Using {dur_sec} seconds for traversing dataset")


if __name__ == "__main__":
    sim()
