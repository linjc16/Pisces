
TASK=binary_class_task
ARCH=drug_dv_base
CLSHEAD=bclsmlpdvppi
CRITERION=binary_class_loss_bce
DATAFOLD=$1
LR=$2
DROP=0.1
MEMORY=32

DATADIR=/data/linjc/dds/data/transductive/$DATAFOLD/data-bin
SAVEDIR=/data/linjc/dds/ckpt/$TASK/$ARCH/$CRITERION/$DATAFOLD/$CLSHEAD/baseline_ljc_lr$LR-norm-drop$DROP-memory$MEMORY-v2


CUDA_VISIBLE_DEVICES=0 python dds/src/binary_class_inf.py $DATADIR \
    --restore-file $SAVEDIR/checkpoint_best.pt \
    --user-dir dds/src \
    --ddp-backend=legacy_ddp \
    --reset-dataloader \
    -s 'a' -t 'b' \
    --datatype 'tg' \
    --task $TASK \
    --arch $ARCH \
    --criterion $CRITERION \
    --max-positions 512 \
    --batch-size 128 \
    --n-memory $MEMORY \
    --optimizer adam \
    --classification-head-name $CLSHEAD \
    --num-classes 125 \
    --fp16 \
    --no-progress-bar \
    --valid-subset 'valid' \
    --skip-update-state-dict \


# TASK=inductive_task
# ARCH=drug_transfomer_large
# CLSHEAD=bclsmlp
# CRITERION=binary_class_loss
# DATAFOLD=$1
# DROP=$2

# # DATADIR=/home/v-xialiang/blob2_containter/v-xialiang/dmp/data/new_split/inductive/$DATAFOLD/data-bin
# # SAVEDIR=ckpt/bib_ddi_ckpt/ddi_zoo/$TASK/$ARCH/$CRITERION/$DATAFOLD/transformer_raw

# # DATADIR=/blob2_container/v-xialiang/dmp/data/new_split/inductive/$DATAFOLD/data-bin
# # SAVEDIR=/blob2_container/v-xialiang/dmp/ckpt/bib_ddi_ckpt/ddi_zoo/$TASK/$ARCH/$CRITERION/$DATAFOLD/transformer_raw

# DATADIR=/home/v-xialiang/blob2_containter/v-xialiang/dmp/data/new_split/ind_unseen/$DATAFOLD/data-bin
# SAVEDIR=ckpt/bib_ddi_ckpt/ddi_zoo/$TASK/$ARCH/$CRITERION/$DATAFOLD/ind_unseen/transformer_raw

# # DATADIR=/blob2_container/v-xialiang/dmp/data/new_split/ind_unseen/$DATAFOLD/data-bin
# # SAVEDIR=/blob2_container/v-xialiang/dmp/ckpt/bib_ddi_ckpt/ddi_zoo/$TASK/$ARCH/$CRITERION/$DATAFOLD/ind_unseen/transformer_raw


# CUDA_VISIBLE_DEVICES=0 python ddi_zoo/src/binary_class_inf.py $DATADIR \
#     --restore-file $SAVEDIR/checkpoint_best.pt \
#     --user-dir ddi_zoo/src/ \
#     -s 'a' -t 'b' \
#     --task $TASK \
#     --arch $ARCH \
#     --criterion $CRITERION \
#     --max-positions 512 \
#     --batch-size 512 \
#     --optimizer adam \
#     --classification-head-name $CLSHEAD \
#     --num-classes 86 \
#     --fp16 \
#     --no-progress-bar \
#     --valid-subset 'valid' \

