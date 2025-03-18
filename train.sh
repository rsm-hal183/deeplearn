#!/bin/bash
# ����CUDA�豸
export CUDA_VISIBLE_DEVICES=0,1
# ����OpenMP�߳���
export OMP_NUM_THREADS=8  # ����CPU���������ã�ͨ������Ϊ�ܺ�����/GPU����
# ����PyTorch��ػ�������
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32
# ʹ��torchrun�����ֲ�ʽѵ��
torchrun --nproc_per_node=2 --master_port=29500 train.py 