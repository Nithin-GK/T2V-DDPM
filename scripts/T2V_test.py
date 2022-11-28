"""
Train a super-resolution model.
"""

import argparse

import torch.nn.functional as F
from core.wandb_logger import WandbLogger
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from torch.utils.data import DataLoader
from guided_diffusion.valdata import  ValData
import os
import torch.distributed as dist
import clip
from guided_diffusion.test_diff import diffusion_test
def main(run):
    args = create_argparser().parse_args()
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"  # set to DETAIL for runtime logging.

    dist_util.setup_dist()
    if(dist.get_rank()==0):

        logger.configure(dir='./experiments/log/')
    if(dist.get_rank()==0):
        logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    model_weights=args.weights

    model.convert_to_fp16()
    model.load_state_dict(
        dist_util.load_state_dict(model_weights, map_location="cpu")
    )
    model.eval()

    val_data = DataLoader(ValData(args.data_dir), batch_size=1, shuffle=False, num_workers=1)  #load_superres_dataval()
    diffusion_test(val_data,model,diffusion, './results/', run , 'test', skip_timesteps=40, iter=0)


def create_argparser():
    defaults = dict(
        data_dir='./data/test/TH/',
        weights="./weights/latest.pt",
        use_fp16=False,

    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser
if __name__ == "__main__":
    run=WandbLogger()
    main(run)
