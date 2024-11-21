
# miAffine
CUDA_VISIBLE_DEVICES=1 python certify_wm.py --base_model miweights/CelebA/stega/miaffine_0.01/epoch_99_state.pth --sigma 0.01 --skip 1 --max 500 --certify_batch_sz 500 --N 10000 --certify_method affine --experiment_name miaffine_0.01 --dataset ae --model stega &
CUDA_VISIBLE_DEVICES=6 python certify_wm.py --base_model miweights/CelebA/stega/miaffine_0.02/epoch_99_state.pth --sigma 0.02 --skip 1 --max 500 --certify_batch_sz 500 --N 10000 --certify_method affine --experiment_name miaffine_0.02 --dataset ae --model stega &
CUDA_VISIBLE_DEVICES=3 python certify_wm.py --base_model miweights/CelebA/stega/miaffine_0.03/epoch_99_state.pth --sigma 0.03 --skip 1 --max 500 --certify_batch_sz 500 --N 10000 --certify_method affine --experiment_name miaffine_0.03 --dataset ae --model stega

