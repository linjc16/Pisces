
TASK=binary_class_task
ARCH=drug_gcn_base
CLSHEAD=bclsmlpppiv2
CRITERION=binary_class_loss_bce
DATAFOLD=$1
LR=$2
DROP=0.1
MEMORY=128

DATADIR=/data/linjc/dds/data/leave_cells/$DATAFOLD/data-bin
SAVEDIR=/data/linjc/dds/ckpt_leave_cells/$TASK/$ARCH/$CRITERION/$DATAFOLD/$CLSHEAD/baseline_ljc_lr$LR-norm-drop$DROP-memory$MEMORY

CUDA_VISIBLE_DEVICES=7 python dds/src/binary_class_inf.py $DATADIR \
    --restore-file $SAVEDIR/checkpoint45.pt \
    --user-dir dds/src \
    --reset-dataloader \
    -s 'a' -t 'b' \
    --task $TASK \
    --arch $ARCH \
    --criterion $CRITERION \
    --max-positions 512 \
    --batch-size 512 \
    --n-memory $MEMORY \
    --optimizer adam \
    --gnn-norm layer \
    --classification-head-name $CLSHEAD \
    --num-classes 125 \
    --fp16 \
    --no-progress-bar \
    --valid-subset 'valid' \
