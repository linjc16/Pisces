
TASK=binary_class_task
ARCH=drug_transfomer_base
CLSHEAD=bclsmlpv2
CRITERION=binary_class_loss_bce
DATAFOLD=$1
LR=$2
DROP=$3

DATADIR=/data/linjc/dds/data/transductive/$DATAFOLD/data-bin
SAVEDIR=/data/linjc/dds/ckpt/$TASK/$ARCH/$CRITERION/$DATAFOLD/$CLSHEAD/baseline_ljc_lr$LR-norm-drop$DROP

# rm -rf $SAVEDIR
mkdir -p $SAVEDIR

CUDA_VISIBLE_DEVICES=7 python dds/src/train.py $DATADIR \
    --user-dir dds/src/ \
    --tensorboard-logdir $SAVEDIR \
    --ddp-backend=legacy_ddp \
    -s 'a' -t 'b' \
    --task $TASK \
    --arch $ARCH \
    --criterion $CRITERION \
    --max-positions 512 \
    --batch-size 128 \
    --update-freq 1 \
    --required-batch-size-multiple 1 \
    --classification-head-name $CLSHEAD \
    --num-classes 125 \
    --dropout $DROP --attention-dropout $DROP --pooler-dropout 0.1 \
    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR \
    --warmup-updates 4000 --total-num-update 100000  --max-update 100000 \
    --log-format 'simple' --log-interval 100 \
    --fp16 \
    --best-checkpoint-metric auc_prc --maximize-best-checkpoint-metric \
    --shorten-method "truncate" \
    --find-unused-parameters \
    --save-dir $SAVEDIR | tee -a $SAVEDIR/train.log \
