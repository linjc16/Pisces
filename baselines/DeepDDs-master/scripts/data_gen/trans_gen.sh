CELLDIR=baselines/DeepDDs-master/data_ours/cell_features_expression_new.csv
SAVEDIR=/data/linjc/dds/baselines/DeepDDS/data_trans_new/
ROOT=baselines/DeepDDs-master/data_ours/

python baselines/DeepDDs-master/creat_data_DC.py \
       --cellfile $CELLDIR \
       --savedir $SAVEDIR \
       --root $ROOT