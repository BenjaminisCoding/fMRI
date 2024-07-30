import hydra
from omegaconf import DictConfig, OmegaConf
from dataloader_fastmri import FastMRIIPR
from model_MRI import GradientDescent, FISTA, PnPMRI
from physic import Nufft


def obtain_data(cfg: DictConfig):
    dataset = FastMRIIPR(
        dataset_path=cfg.DATASET_PATH, 
        test=cfg.TEST, 
        challenge=cfg.CHALLENGE,
        acceleration_factor=cfg.ACCELERATION_FACTOR,
        density=cfg.DENSITY,
        trajectory=cfg.TRAJECTORY,
        combine_method = cfg.COMBINE_METHOD,
        var_noise = cfg.VAR_NOISE,
        filter_size_smaps = cfg.FILTER_SIZE_SMAPS
        )
    kspace, image, target, y, smaps, mask, physic, physic_multicoil = dataset.__getitem__(cfg.DATA_ID)
    return kspace, image, target, y, smaps, mask, physic, physic_multicoil

@hydra.main(version_base=None, config_path="config", config_name="config_optuna")
def run_algorithm(cfg: DictConfig):
    kspace, image, target, y, smaps, mask, physic, physic_multicoil = obtain_data(cfg['data'])
    set_up_data = {
        'physic': physic_multicoil,
        'y': y,
        'target': target,
        'mask': mask,
    }
    assert cfg.MODEL_NAME in ['GD', 'FISTA', 'PnP']
    if cfg.MODEL_NAME == 'GD':
        model = GradientDescent(**set_up_data , **cfg)
    elif cfg.MODEL_NAME == 'FISTA':
        model = FISTA(**set_up_data , **cfg)
    elif cfg.MODEL_NAME == 'PnP':
        model = PnPMRI(**set_up_data , **cfg)
    else:
        raise ValueError(f"Unknown model type: {cfg['model'].MODEL_NAME}")
    _, L_PSNR, _ = model.run()
    return max(L_PSNR)

if __name__ == '__main__':
    run_algorithm()
