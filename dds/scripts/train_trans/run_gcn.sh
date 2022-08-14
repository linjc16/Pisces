
TASK=binary_class_task
ARCH=drug_gcn_large
CLSHEAD=bclsmlpv2
CRITERION=binary_class_loss
DATAFOLD=$1

DATADIR=/data/linjc/dds/data/transductive/$DATAFOLD/data-bin
SAVEDIR=/data/linjc/dds/ckpt/$TASK/$ARCH/$CRITERION/$DATAFOLD/$CLSHEAD/baseline_ljc_2

# rm -r $SAVEDIR
mkdir -p $SAVEDIR

CUDA_VISIBLE_DEVICES=0 python dds/src/train.py $DATADIR \
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
    --pooler-dropout 0.1 \
    --gnn-norm layer \
    --weight-decay 1e-4 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr 1e-4 --total-num-update 100000 \
    --warmup-updates 4000 --max-update 100000 \
    --log-format 'simple' --log-interval 100 \
    --fp16 \
    --best-checkpoint-metric roc_auc --maximize-best-checkpoint-metric \
    --shorten-method "truncate" \
    --find-unused-parameters \
    --save-dir $SAVEDIR | tee -a $SAVEDIR/train.log \