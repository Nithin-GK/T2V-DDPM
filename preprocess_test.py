
import math
import numpy as np
import torch as th
from PIL import Image
import argparse 
import os

def process_thermal(thermal_dir, vis_img,eps,dest_dir):
    
        if(os.path.exists(dest_dir)==False):
            os.makedirs(dest_dir)
        thermal_images=os.listdir(thermal_dir)
        pil_image_vis =  Image.open(vis_img)
        pil_image_vis =pil_image_vis.resize((128,128))
        arr_vis=np.array(pil_image_vis)/255.0
        for _ in thermal_images:
            pil_image_th = Image.open(os.path.join(thermal_dir,_))
            pil_image_th =pil_image_th.resize((128,128))

        # --- Transform to tensor --- #
            arr_th=np.array(pil_image_th)/255.0
            arr_visible=arr_vis.copy()
            arr_mask = arr_th.copy()
            arr_mask[arr_th>0.1]=1
            arr_mask[arr_th<=0.1]=0

            th_process= arr_mask*arr_th + (1-arr_mask)*arr_visible

            th_process=np.uint8((th_process*255.0))
            th_process = Image.fromarray(th_process)
            th_process.save(os.path.join(dest_dir,_))
            
        
def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

        
def create_argparser():
    defaults = dict(
        thermal_dir='./data/test/TH/',
        visible_sample="./data/test/sample.png",
        dest_dir='./data/test_process/TH/',
        eps=0.1,
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser       

if __name__ == "__main__":
    args = create_argparser().parse_args()
    thermal_dir=args.thermal_dir
    visible_image=args.visible_sample
    dest_dir=args.dest_dir
    process_thermal(thermal_dir,visible_image,args.eps,dest_dir)
