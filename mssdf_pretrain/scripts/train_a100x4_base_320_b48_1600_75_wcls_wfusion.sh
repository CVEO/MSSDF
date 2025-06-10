#!/bin/bash
#SBATCH -p a100x4
#SBATCH --gres=gpu:4
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1

script_name_no_suffix=${SLURM_JOB_NAME%.sh}
OUTPUT_DIR="output/${script_name_no_suffix}"
DATA_PATH='< path to dataset >'

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

# batch_size can be adjusted according to the graphics card
OMP_NUM_THREADS=1 torchrun --nnodes=4 --nproc_per_node=4 run_mae_pretraining_mm.py \
        --data_path ${DATA_PATH} \
        --mask_ratio 0.75 \
        --input_size 320 \
        --model pretrain_mae_base_patch16 \
        --batch_size 32 \
        --opt adamw \
        --fusion_depth 3 \
        --cls_token \
        --ce_weight 0.1 \
        --drop_path 0 \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --epochs 500 \
        --output_dir ${OUTPUT_DIR}