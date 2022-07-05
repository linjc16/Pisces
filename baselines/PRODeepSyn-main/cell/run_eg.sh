
python baselines/PRODeepSyn-main/cell/train.py target_ge.npy nodes_ge.npy \
    --suffix 'sample' \
    --gpu 0 \
    --emb_dim 384 \
    --batch 2 \
    --lr 1e-3