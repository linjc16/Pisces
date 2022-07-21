
DATAFOLD=$1
SAVEDIR=/data/linjc/dds/data/transductive_3fold/$DATAFOLD


python dds/scripts/data_process/run_process_trans.py \
    --output_dir $SAVEDIR \

