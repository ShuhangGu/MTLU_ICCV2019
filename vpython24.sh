#!/bin/bash
#$ -l gpu=1
#$ -l h_rt=24:00:00
#$ -l h_vmem=40G
#$ -cwd
#$ -V
#$ -j y
#$ -o /scratch_net/amaia/shgu/github/MTLU/Checkpoints

echo "$@"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python $@


# -l h="bmicgpu0[3-5]|biwirender1[3-9]|bmicgpu02|bmicgpu01|biwirender1[0-2]|biwirender0[5-9]"

