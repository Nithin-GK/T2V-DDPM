# 3SD
T2V-DDPM: Thermal to Visible Face Translation using Denoising Diffusion Probabilistic Models

[Paper link](https://arxiv.org/pdf/2209.08814.pdf)

Modern-day surveillance systems perform person recognition using deep learning-based face verification networks. Most state-of-the-art facial verification systems are
trained using visible spectrum images. But, acquiring images in the visible spectrum is impractical in scenarios of low-light and nighttime conditions, and often images are captured in an alternate domain such as the thermal infrared domain. Facial verification in thermal images is often performed after retrieving the corresponding visible domain images. This is a well-established problem often known as the Thermal-to-Visible (T2V) image translation. In this paper, we propose a Denoising Diffusion Probabilistic Model (DDPM) based solution for T2V translation specifically for facial images. During training, the model learns the conditional distribution of visible facial images given their corresponding thermal image through the diffusion process. During inference, the visible domain image is obtained by starting from Gaussian noise and performing denoising repeatedly. The existing inference process for DDPMs is stochastic and time-consuming. Hence, we propose a novel inference strategy for speeding up the inference time of DDPMs, specifically for the problem of T2V image translation. We achieve the state-of-the-art results on multiple datasets. 

## Prerequisites:
1. Create a conda environment and activate using 
```
conda env create -f environment.yml
conda activate T2V-diff
```
## Data Preparation
2. Prepare Data in the following format
```
    ├── data 
    |   ├── train # Training  
    |   |   ├── TH              # thermal images 
    |   |   └── VIS             # visible images
    |   └── test  # Testing
    |       ├── TH              # thermal images 
```
## Training and Testing
3 Run following commands to train and test 
```
For training:
export PYTHONPATH=$PYTHONPATH:$(pwd)
CUDA_VISIBLE_DEVICES="0" NCCL_P2P_DISABLE=1  torchrun --nproc_per_node=1 --master_port=4326 scripts/T2V_train.py 

For testing:
export PYTHONPATH=$PYTHONPATH:$(pwd)
CUDA_VISIBLE_DEVICES="0" NCCL_P2P_DISABLE=1  torchrun --nproc_per_node=1 --master_port=4326 scripts/T2V_test.py --weights /pathtoweights/ --data_dir /pathtodata/
```
4. If you use our work, please use the following citation
```
@article{nair2022t2v,
  title={T2V-DDPM: Thermal to Visible Face Translation using Denoising Diffusion Probabilistic Models},
  author={Nair, Nithin Gopalakrishnan and Patel, Vishal M},
  journal={arXiv preprint arXiv:2209.08814},
  year={2022}
}
```

## Acknowledgements
Thanks to authors of Diffusion Models Beat GANs on Image Synthesis sharing their code. Most of the code is borrowed from the guided diffusion
```
https://github.com/openai/guided-diffusion
```
