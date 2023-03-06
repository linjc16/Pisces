SAVEDIR=/data/linjc/dds/baselines/DeepSynergy/data_pt2_new
RAW_DATA_DIR=baselines/DeepDDs-master/data_ours

python baselines/DeepSynergy-master/preprocessing/data_generate_tpm.py \
    --savedir $SAVEDIR \
    --raw_data_dir $RAW_DATA_DIR