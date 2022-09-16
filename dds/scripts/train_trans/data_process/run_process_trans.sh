
DATAFOLD=$1
SAVEDIR=/data/linjc/dds/data_new/transductive/$DATAFOLD


python dds/scripts/train_trans/data_process/run_process_trans.py \
    --output_dir $SAVEDIR \

