# #!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)

CUDA_VISIBLE_DEVICES="0" NCCL_P2P_DISABLE=1  torchrun --nproc_per_node=1 --master_port=4326 scripts/T2V_train.py 