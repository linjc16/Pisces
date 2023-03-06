FOLD=$1
DATADIR=/data/linjc/dds/baselines/DeepSynergy/data_leave_cell_new

CUDA_VISIBLE_DEVICES=0 python baselines/DeepSynergy-master/train.py --fold $FOLD --datadir $DATADIR