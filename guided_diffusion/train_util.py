import copy
import functools
import os

import blobfile as bf
import torch as th
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import cv2
import torch.distributed as dist

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .resample import LossAwareSampler, UniformSampler
import numpy as np
# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0
import tqdm 
from tqdm import tqdm
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from .test_diff import diffusion_test
import clip

# import clip
class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        val_dat,
        batch_size,
        microbatch,
        lr,
        log_interval,
        save_interval,
        test_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.val_data=val_dat
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr

        self.test_interval = test_interval
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = None or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                dict_load = dist_util.load_state_dict(resume_checkpoint, map_location=dist_util.dev())
                self.model.load_state_dict(dict_load, strict=False)
              
        dist_util.sync_params(self.model.parameters())


    def test(self, run, phase='test',skip_timesteps=0,iter=0):
        diffusion_test( self.val_data, self.model,self.diffusion,'./results', run, phase, skip_timesteps,iter )


    def run_loop(self, run):
        num_iter=100000

        for i in tqdm(range(num_iter)):
            batch, cond = next(self.data)
            self.run_step(batch, cond)

            # if self.step % self.log_interval == 0:
            #         run.log_metrics(logger.getkvs())

            if (self.step + 1) % self.save_interval == 0:
                self.save()
                
            if (self.step) % self.test_interval == 0:
                self.test(run, phase='train', skip_timesteps=0, iter = i)

            self.step += 1


    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)

        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            # print("hi")
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)


    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        if(dist.get_rank()==0):
            logger.logkv("step", self.step + self.resume_step)
            logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint( params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model")
                filename = f"model{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join("./weights", filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(self.mp_trainer.master_params)


        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()



def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        if(dist.get_rank()==0):

            logger.logkv_mean(key, values.mean().item())
            # Log the quantiles (four quartiles, in particular).
            for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
                quartile = int(4 * sub_t / diffusion.num_timesteps)
                logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
