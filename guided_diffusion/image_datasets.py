import math
import random
import torch as th
from PIL import Image
import blobfile as bf
# from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset
import cv2
import torch.distributed as dist
import os

def load_data(
    *,
    data_dir,
    gt_dir,
    batch_size,
    image_size,
    deterministic=False,
    random_flip=True,
):
    """
    For a dataset, create a generator over (images, kwargs) pairs.

    Each images is an NCHW float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for class labels, in which case the key is "y"
    and the values are integer tensors of class labels.

    :param data_dir: a dataset directory.
    :param batch_size: the batch size of each returned pair.
    :param image_size: the size to which images are resized.
    :param class_cond: if True, include a "y" key in returned dicts for class
                       label. If classes are not available and this is true, an
                       exception will be raised.
    :param deterministic: if True, yield results in a deterministic order.
    :param random_crop: if True, randomly crop the images for augmentation.
    :param random_flip: if True, randomly flip the images for augmentation.
    """
    if not data_dir:
        raise ValueError("unspecified data directory")
    img_files =os.listdir(gt_dir)

    dataset = ImageDataset(
        image_size,
        img_files,
        data_dir,
        gt_dir,
        shard=dist.get_rank(),
        num_shards=dist.get_world_size(),
        random_flip=random_flip,
    )
    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=1, drop_last=True
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=1, drop_last=True
        )
    while True:
        yield from loader


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results


class RandomCrop(object):

    def __init__(self, crop_size=[256,128]):
        """Set the height and weight before and after cropping"""
        self.crop_size_h  = crop_size[0]
        self.crop_size_w  = crop_size[1]

    def __call__(self, inputs, target):
        input_size_h, input_size_w, _ = inputs.shape
        try:
            x_start = random.randint(0, input_size_w - self.crop_size_w)
            y_start = random.randint(0, input_size_h - self.crop_size_h)
            inputs = inputs[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] 
            target = target[y_start: y_start + self.crop_size_h, x_start: x_start + self.crop_size_w] 
        except:
            inputs=cv2.resize(inputs,(128,128))
            target=cv2.resize(target,(128,128))

        return inputs,target

class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        image_paths,
        data_dir,
        gt_paths,
        shard=0,
        num_shards=1,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = image_paths[shard:][::num_shards]
        self.random_flip = random_flip
        self.gt_paths=gt_paths
        self.data_dir=data_dir


    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir,self.local_images[idx])
        gt_path = os.path.join(self.gt_paths,self.local_images[idx])

        with bf.BlobFile(path, "rb") as f:
            thermal_image = self.process_and_load_images(path)
        with bf.BlobFile(gt_path, "rb") as f1:
            visible_image = self.process_and_load_images(gt_path)


        out_dict = {}
        out_dict["thermal"]=thermal_image
        out_dict["visible"]=visible_image
        return visible_image, out_dict
        
    def process_and_load_images(self,path):
        pil_image = Image.open(path)
        pil_image.load()
        pil_image=pil_image.resize((self.resolution,self.resolution))
        arr=np.array(pil_image).astype(np.float32)
        arr=arr/127.5-1.0
        arr = np.transpose(arr, [2, 0, 1])

        return arr

