
TASK=binary_class_task
ARCH=drug_gcn_large
CLSHEAD=bclsmlpv2
CRITERION=binary_class_loss_bce
DATAFOLD=$1
LR=$2
DROP=$3

DATADIR=/data/linjc/dds/data/transductive/$DATAFOLD/data-bin
SAVEDIR=/data/linjc/dds/ckpt/$TASK/$ARCH/$CRITERION/$DATAFOLD/$CLSHEAD/baseline_ljc_lr$LR-norm-drop$DROP

# rm -r $SAVEDIR
mkdir -p $SAVEDIR

CUDA_VISIBLE_DEVICES=1 python dds/src/train.py $DATADIR \
    --user-dir dds/src/ \
    -s 'a' -t 'b' \
    --tensorboard-logdir $SAVEDIR \
    --task $TASK \
    --arch $ARCH \
    --criterion $CRITERION \
    --max-positions 512 \
    --batch-size 128 \
    --required-batch-size-multiple 1 \
    --classification-head-name $CLSHEAD \
    --num-classes 125 \
    --pooler-dropout $DROP \
    --gnn-norm layer \
    --weight-decay 1e-5 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update 150000 \
    --warmup-updates 9000 --max-update 150000 \
    --log-format 'simple' --log-interval 100 \
    --fp16 \
    --best-checkpoint-metric roc_auc --maximize-best-checkpoint-metric \
    --shorten-method "truncate" \
    --find-unused-parameters \
    --save-dir $SAVEDIR | tee -a $SAVEDIR/train.log \