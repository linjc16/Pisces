
TASK=binary_class_task
ARCH=drug_gcn_tiny
CLSHEAD=bclsmlpppi
CRITERION=binary_class_loss_bce
DATAFOLD=$1
LR=$2
DROP=0.1

DATADIR=/data/linjc/dds/data/transductive/$DATAFOLD/data-bin
SAVEDIR=/data/linjc/dds/ckpt/$TASK/$ARCH/$CRITERION/$DATAFOLD/$CLSHEAD/baseline_ljc_lr$LR-norm-drop$DROP-new

CUDA_VISIBLE_DEVICES=0 python dds/src/binary_class_inf.py $DATADIR \
    --restore-file $SAVEDIR/checkpoint_best.pt \
    --user-dir dds/src \
    --reset-dataloader \
    -s 'a' -t 'b' \
    --task $TASK \
    --arch $ARCH \
    --criterion $CRITERION \
    --max-positions 512 \
    --batch-size 512 \
    --optimizer adam \
    --gnn-norm layer \
    --classification-head-name $CLSHEAD \
    --num-classes 125 \
    --fp16 \
    --no-progress-bar \
    --valid-subset 'valid' \
