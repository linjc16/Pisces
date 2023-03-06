CELLDIR=baselines/DeepDDs-master/data_ours_leave_comb/cell_features_expression_new.csv
SAVEDIR=/data/linjc/dds/baselines/DeepDDS/data_leave_comb_new/
ROOT=baselines/DeepDDs-master/data_ours_leave_comb/

python baselines/DeepDDs-master/creat_data_DC.py \
       --cellfile $CELLDIR \
       --savedir $SAVEDIR \
       --root $ROOT