FOLD=$1
DATADIR=/data/linjc/dds/baselines/DeepDDS/data_trans_new


CUDA_VISIBLE_DEVICES=4 python baselines/DeepDDs-master/training_GCN.py --fold $FOLD --datadir $DATADIR