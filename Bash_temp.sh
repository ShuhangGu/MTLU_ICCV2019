#!/bin/bash
#Submit to GPU

##  Train FDmtlu
python main.py  --model fdmtlu --m_blocks 10 --n_feats 64 --bin_num 40 --bin_width 0.05 --bn True --noise_sigma 50 --scale 1 --n_colors 1 --lr 1e-3 --gamma 0.5 --lr_decay 20  --test_every 10000  --patch 128 --batch 72 --ext bin --data_train DIV2KDENOISE --data_test DenoiseSet68  --testbin True  --save ./Checkpoints/FDmtlu10_Sigma50_B40_F64_P128B72

## Train FSRmtlu
#python main.py  --model fsrmtlu --m_blocks 7 --n_feats 64 --bin_num 40 --bin_width 0.05 --bn True --noise_sigma 0 --scale 4 --n_colors 3 --lr 1e-3 --gamma 0.5 --lr_decay 20  --test_every 10000  --patch 128 --batch 72 --ext bin --data_train DIV2K --data_test Set5_x4  --testbin True  --save ./Checkpoints/FSRmtlu7_Scale4_B40_F64_P128B72

# Test FDmtlu
#python main.py  --model fdmtlu --m_blocks 10 --n_feats 64 --bin_num 40 --bin_width 0.05 --bn True --noise_sigma 50 --scale 1 --n_colors 1 --lr 1e-3 --gamma 0.5 --lr_decay 20  --test_every 10000  --patch 128 --batch 72 --ext bin --data_train DIV2KDENOISE --data_test DenoiseSet68  --testbin True --pre_train ./Checkpoints/FDmtlu10_Sigma50_B40_F64_P128B72/model/model_latest.pt --test_only

# Test FSRmtlu
#python main.py  --model fdmtlu --m_blocks 10 --n_feats 64 --bin_num 40 --bin_width 0.05 --bn True --noise_sigma 50 --scale 1 --n_colors 1 --lr 1e-3 --gamma 0.5 --lr_decay 20  --test_every 10000  --patch 128 --batch 72 --ext bin --data_train DIV2KDENOISE --data_test DenoiseSet68  --testbin True --pre_train ./Checkpoints/FSRmtlu7_Scale4_B40_F64_P128B72/model/model_latest.pt --test_only
