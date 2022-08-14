
TASK=binary_class_task
ARCH=drug_transfomer_large
CLSHEAD=bclsmlpppiv2
CRITERION=binary_class_loss_bce
DATAFOLD=$1
LR=$2
DROP=0.3
MEMORY=128

DATADIR=/data/linjc/dds/data/transductive/$DATAFOLD/data-bin
SAVEDIR=/data/linjc/dds/ckpt/$TASK/drug_dv_large/$CRITERION/$DATAFOLD/$CLSHEAD/baseline_ljc_lr$LR-norm-drop$DROP-memory$MEMORY

CUDA_VISIBLE_DEVICES=0 python dds/src/binary_class_inf.py $DATADIR \
    --restore-file $SAVEDIR/checkpoint103.pt \
    --reset-dataloader \
    --user-dir dds/src/ \
    -s 'a' -t 'b' \
    --task $TASK \
    --arch $ARCH \
    --criterion $CRITERION \
    --max-positions 512 \
    --batch-size 128 \
    --optimizer adam \
    --classification-head-name $CLSHEAD \
    --num-classes 125 \
    --n-memory $MEMORY \
    --fp16 \
    --no-progress-bar \
    --valid-subset 'valid' \
