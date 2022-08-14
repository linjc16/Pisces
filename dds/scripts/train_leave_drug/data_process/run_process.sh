
DATAFOLD=$1
SAVEDIR=/data/linjc/dds/data/leave_drugs/$DATAFOLD


python dds/scripts/train_leave_drug/data_process/run_process.py \
    --output_dir $SAVEDIR \

