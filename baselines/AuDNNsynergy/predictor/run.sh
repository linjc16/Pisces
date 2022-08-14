CUDA_VISIBLE_DEVICES=7 python baselines/AuDNNsynergy/predictor/cross_validation.py \
    --gpu 0 \
    --hidden 2048 \
    --lr 1e-3 1e-4 1e-5
    --batch 512