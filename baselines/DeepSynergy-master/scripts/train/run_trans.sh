FOLD=$1
DATADIR=/data/linjc/dds/baselines/DeepSynergy/data_pt2_new

CUDA_VISIBLE_DEVICES=3 python baselines/DeepSynergy-master/train.py --fold $FOLD --datadir $DATADIR