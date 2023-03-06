CELLDIR=baselines/DeepDDs-master/data_ours_leave_cell/cell_features_expression_new.csv
SAVEDIR=/data/linjc/dds/baselines/DeepDDS/data_leave_cell_new/
ROOT=baselines/DeepDDs-master/data_ours_leave_cell/

python baselines/DeepDDs-master/creat_data_DC.py \
       --cellfile $CELLDIR \
       --savedir $SAVEDIR \
       --root $ROOT