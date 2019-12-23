#!/bin/bash
#Submit to GPU




for SIGMA in 50; do
MODEL=fdmtlu
M_BLOCK=10
FEAT=64
LR=6e-5
GAMMA=0.5
LR_DECAY=20
PATCH=128
BATCH=72
TEST_EVE=10000
BIN_NUM=10
BIN_WIDTH=0.2
BN=True
qsub ./vpython24.sh ./main.py  --model ${MODEL} --m_blocks ${M_BLOCK}  --n_feats ${FEAT} --bin_num ${BIN_NUM} --bin_width ${BIN_WIDTH} --bn ${BN} \
--noise_sigma ${SIGMA} --scale 1 --n_colors 1 \
--lr ${LR} --gamma ${GAMMA} --lr_decay ${LR_DECAY}  --test_every ${TEST_EVE}  --patch ${PATCH} --batch ${BATCH}  \
--ext bin --data_train DIV2KDENOISE --data_test DenoiseSet68  --testbin True \
--save ./Checkpoints/${MODEL}${M_BLOCK}_Sigma${SIGMA}_${BIN_NUM}B_F${FEAT}_p${PATCH}_b${BATCH} \
--pre_train ./Checkpoints/${MODEL}${M_BLOCK}_Sigma${SIGMA}_${BIN_NUM}B_F${FEAT}_p${PATCH}_b${BATCH}/model/model_best.pt
done


