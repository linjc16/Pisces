FOLD=$1
DATADIR=/data/linjc/dds/baselines/DeepSynergy/data_leave_comb_new

CUDA_VISIBLE_DEVICES=5 python baselines/DeepSynergy-master/train.py --fold $FOLD --datadir $DATADIR