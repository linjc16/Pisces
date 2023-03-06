FOLD=$1

python baselines/PRODeepSyn-main/predictor/validation_cell.py --batch 512 --hidden 2048 4096 8192 --lr 0.001 0.0001 0.00001 --gpu 0 --suffix fold$FOLD