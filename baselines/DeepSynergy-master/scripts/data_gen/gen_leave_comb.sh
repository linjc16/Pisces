SAVEDIR=/data/linjc/dds/baselines/DeepSynergy/data_leave_comb_new
RAW_DATA_DIR=baselines/DeepDDs-master/data_ours_leave_comb

python baselines/DeepSynergy-master/preprocessing/data_generate_tpm.py \
    --savedir $SAVEDIR \
    --raw_data_dir $RAW_DATA_DIR