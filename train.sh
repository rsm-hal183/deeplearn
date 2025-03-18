#!/bin/bash
# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=0,1
# 设置OpenMP线程数
export OMP_NUM_THREADS=8  # 根据CPU核心数设置，通常设置为总核心数/GPU数量
# 设置PyTorch相关环境变量
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
# 使用torchrun启动分布式训练
torchrun --nproc_per_node=2 --master_port=29500 train.py 