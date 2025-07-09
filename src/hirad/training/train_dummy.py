import hydra
from omegaconf import DictConfig, OmegaConf
import json


@hydra.main(version_base=None, config_path="../conf", config_name="training")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    cfg = OmegaConf.to_container(cfg)
    print(json.dumps(cfg, indent=2))
    # print(cfg.pretty())

if __name__ == "__main__":
    main()