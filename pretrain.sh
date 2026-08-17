#!/usr/bin/env bash
# CRCME two-stage self-supervised pre-training.
# Run Stage I first, then pass the resulting checkpoints to Stage II via
# --ct_pretrain_model and --wsi_pretrain_model.

# ---- Stage I: single-modality masked autoencoding (CT and pathology) ----
CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 --master_port 23456 \
    src/pretrain_unimodal.py \
    --world_size 2 \
    --mask_ratio 0.75

# ---- Stage II: pathology-guided distillation (frozen teacher -> Joint Expert) ----
CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 --master_port 23456 \
    src/pretrain_KD.py \
    --world_size 2 \
    --mask_ratio 0.75 \
    --ct_pretrain_model  /path/to/stage1_ct_checkpoint.pth \
    --wsi_pretrain_model /path/to/stage1_pathology_checkpoint.pth