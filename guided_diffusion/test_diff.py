

import blobfile as bf
import torch as th
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW
import cv2
import torch.distributed as dist

from . import dist_util
import numpy as np
import os
from tqdm import tqdm
def diffusion_test(val_data,model,diffusion, save_dir, run , phase, skip_timesteps=0, iter=0):
               
                    model.eval()
                    if(dist.get_rank()==0):
                        save_fold =os.path.join(save_dir, phase)

                        if(os.path.exists(save_fold)==False):
                            if(os.path.exists(save_fold)==False):
                                os.makedirs(save_fold)

                        if(phase=='train'):
                            save_fold=os.path.join(save_fold, 'iter_'+ str(iter))
                            if(os.path.exists(save_fold)==False):
                                os.makedirs(save_fold)

                    with th.no_grad():
                                for batch_id, data_dict in enumerate(val_data):
                                    model_kwargs={}
                                
                                    for k, v in data_dict.items():
                                        if('Index' in k):
                                            img_name=v
                                        else:
                                            model_kwargs[k]= v.to(dist_util.dev())

                                    thermal=model_kwargs['thermal']
                                    timestep = diffusion.num_timesteps - skip_timesteps
                                    device=next(model.parameters()).device
                                    if(skip_timesteps>0):
                                        init_image = diffusion.q_sample(thermal, t=th.tensor(timestep, dtype=th.long, device=device))
                                    else:
                                        init_image=None

                                    sample = diffusion.p_sample_loop(
                                        model,
                                        thermal.shape,
                                        clip_denoised=True,
                                        model_kwargs=model_kwargs,
                                        noise=init_image,
                                        skip_timesteps=skip_timesteps
                                    )

                                    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
                                    sample = sample.permute(0, 2, 3, 1).contiguous().cpu().numpy()
                                    sample = sample
                                    thermal_image = ((model_kwargs['thermal']+1)* 127.5).clamp(0, 255).to(th.uint8)
                                    thermal_image = thermal_image.permute(0, 2, 3, 1).contiguous().cpu().numpy()
                                    if(dist.get_rank()==0):
                                            for i in range(thermal.shape[0]):
                                                img_disp=np.concatenate((thermal_image[i],sample[i]), axis=1)
                                                img_path =os.path.join(save_fold,img_name[i])
                                                cv2.imwrite(img_path,img_disp[:,:,::-1])
