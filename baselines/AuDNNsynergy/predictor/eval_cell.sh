CUDA_VISIBLE_DEVICES=7 python baselines/AuDNNsynergy/predictor/eval_cell.py \
    --gpu 0 \
    --best_hs 2048 \
    --batch 512 \
    --suffix fold0