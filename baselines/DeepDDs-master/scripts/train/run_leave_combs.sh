FOLD=$1
DATADIR=/data/linjc/dds/baselines/DeepDDS/data_leave_comb_new/


CUDA_VISIBLE_DEVICES=6 python baselines/DeepDDs-master/training_GCN.py --fold $FOLD --datadir $DATADIR