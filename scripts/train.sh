# python train.py --cfg ../configs/linear_normal/gcn_linear_normal_test.py --gpu_ids [0]
WANDB_NOTES="$1"
python scripts/train.py --cfg ./configs/linear/train/transformer_linear_lennon7.py --gpu_ids [0] --wandb_notes "$WANDB_NOTES"