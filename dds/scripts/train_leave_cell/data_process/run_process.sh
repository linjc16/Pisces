
DATAFOLD=$1
SAVEDIR=/data/linjc/dds/data_new/leave_cells/$DATAFOLD


python dds/scripts/train_leave_cell/data_process/run_process.py \
    --output_dir $SAVEDIR \

