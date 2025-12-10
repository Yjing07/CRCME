# CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 --master_port 23456 src/pretrain_unimodal.py --world_size 2 ######

CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 --master_port 23456 src/pretrain_KD.py --world_size 2 ######
