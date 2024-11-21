# COCO hidden

# GN
# CUDA_VISIBLE_DEVICES=0 python certify_wm.py --base_model weights/COCO/hidden/GN_0.1/epoch_99_state.pth --sigma 0.1 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name GN_0.1 --dataset coco --model hidden &
# CUDA_VISIBLE_DEVICES=1 python certify_wm.py --base_model weights/COCO/hidden/GN_0.25/epoch_99_state.pth --sigma 0.25 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name GN_0.25 --dataset coco --model hidden &
# CUDA_VISIBLE_DEVICES=2 python certify_wm.py --base_model weights/COCO/hidden/GN_0.5/epoch_99_state.pth --sigma 0.5 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name GN_0.5 --dataset coco --model hidden &

# # Affine
# CUDA_VISIBLE_DEVICES=4 python certify_wm.py --base_model weights/COCO/hidden/affine_0.01/epoch_99_state.pth --sigma 0.01 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name affine_0.01 --dataset coco --model hidden &
# CUDA_VISIBLE_DEVICES=5 python certify_wm.py --base_model weights/COCO/hidden/affine_0.02/epoch_99_state.pth --sigma 0.02 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name affine_0.02 --dataset coco --model hidden &
# CUDA_VISIBLE_DEVICES=6 python certify_wm.py --base_model weights/COCO/hidden/affine_0.03/epoch_99_state.pth --sigma 0.03 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name affine_0.03 --dataset coco --model hidden 

# # Celeba hidden

# CUDA_VISIBLE_DEVICES=0 python certify_wm.py --base_model weights/CelebA/hidden/GN_0.1/epoch_99_state.pth --sigma 0.1 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name GN_0.1 --dataset celeb --model hidden &
# CUDA_VISIBLE_DEVICES=1 python certify_wm.py --base_model weights/CelebA/hidden/GN_0.25/epoch_99_state.pth --sigma 0.25 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name GN_0.25 --dataset celeb --model hidden &
# CUDA_VISIBLE_DEVICES=2 python certify_wm.py --base_model weights/CelebA/hidden/GN_0.5/epoch_99_state.pth --sigma 0.5 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name GN_0.5 --dataset celeb --model hidden &

# # Affine
# CUDA_VISIBLE_DEVICES=4 python certify_wm.py --base_model weights/CelebA/hidden/affine_0.01/epoch_99_state.pth --sigma 0.01 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name affine_0.01 --dataset celeb --model hidden &
# CUDA_VISIBLE_DEVICES=5 python certify_wm.py --base_model weights/CelebA/hidden/affine_0.02/epoch_99_state.pth --sigma 0.02 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name affine_0.02 --dataset celeb --model hidden &
# CUDA_VISIBLE_DEVICES=6 python certify_wm.py --base_model weights/CelebA/hidden/affine_0.03/epoch_99_state.pth --sigma 0.03 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name affine_0.03 --dataset celeb --model hidden 

# Celeba stega

# CUDA_VISIBLE_DEVICES=0 python certify_wm.py --base_model weights/CelebA/stega/GN_0.1/epoch_99_state.pth --sigma 0.1 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method GN --experiment_name GN_0.1 --dataset celeb --model stega &
# CUDA_VISIBLE_DEVICES=1 python certify_wm.py --base_model weights/CelebA/stega/GN_0.25/epoch_99_state.pth --sigma 0.25 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method GN --experiment_name GN_0.25 --dataset celeb --model stega &
# CUDA_VISIBLE_DEVICES=4 python certify_wm.py --base_model weights/CelebA/stega/GN_0.5/epoch_99_state.pth --sigma 0.5 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method GN --experiment_name GN_0.5 --dataset celeb --model stega

# Affine
# CUDA_VISIBLE_DEVICES=4 python certify_wm.py --base_model weights/CelebA/stega/affine_0.01/epoch_99_state.pth --sigma 0.01 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name affine_0.01 --dataset celeb --model stega &
# CUDA_VISIBLE_DEVICES=1 python certify_wm.py --base_model weights/CelebA/stega/affine_0.02/epoch_99_state.pth --sigma 0.02 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name affine_0.02 --dataset celeb --model stega &
# CUDA_VISIBLE_DEVICES=0 python certify_wm.py --base_model weights/CelebA/stega/affine_0.03/epoch_99_state.pth --sigma 0.03 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name affine_0.03 --dataset celeb --model stega 

# COCO stega
# CUDA_VISIBLE_DEVICES=0 python certify_wm.py --base_model weights/COCO/stega/GN_0.1/epoch_99_state.pth --sigma 0.1 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method GN --experiment_name GN_0.1 --dataset coco --model stega &
# CUDA_VISIBLE_DEVICES=1 python certify_wm.py --base_model weights/COCO/stega/GN_0.25/epoch_99_state.pth --sigma 0.25 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method GN --experiment_name GN_0.25 --dataset coco --model stega &
# CUDA_VISIBLE_DEVICES=4 python certify_wm.py --base_model weights/COCO/stega/GN_0.5/epoch_99_state.pth --sigma 0.5 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method GN --experiment_name GN_0.5 --dataset coco --model stega






# # miGN
# CUDA_VISIBLE_DEVICES=4 python certify_wm.py --base_model miweights/COCO/hidden/miGN_0.1/epoch_99_state.pth --sigma 0.1 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name miGN_0.1 --dataset coco --model hidden
# CUDA_VISIBLE_DEVICES=5 python certify_wm.py --base_model miweights/COCO/hidden/miGN_0.25/epoch_99_state.pth --sigma 0.25 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name miGN_0.25 --dataset coco --model hidden 
# CUDA_VISIBLE_DEVICES=2 python certify_wm.py --base_model miweights/COCO/hidden/miGN_0.5/epoch_99_state.pth --sigma 0.5 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name miGN_0.5 --dataset coco --model hidden
# CUDA_VISIBLE_DEVICES=6 python certify_wm.py --base_model miweights/COCO/hidden/miGN_0.5/epoch_99_state.pth --sigma 0.5 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name miGN_0.5 --dataset coco --model hidden





# CUDA_VISIBLE_DEVICES=5 python certify_wm.py --base_model miweights/COCO/hidden/miGN_0.5/epoch_99_state.pth --sigma 0.5 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name miGN_0.5 --dataset coco --model hidden

# # miscaling
# # CUDA_VISIBLE_DEVICES=0 python certify_wm.py --base_model miweights/COCO/hidden/miscaling_uniform_0.1/epoch_99_state.pth --sigma 0.1 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method scaling_uniform --experiment_name miscaling_uniform_0.1 --dataset coco --model hidden &
# # CUDA_VISIBLE_DEVICES=1 python certify_wm.py --base_model miweights/COCO/hidden/miscaling_uniform_0.15/epoch_99_state.pth --sigma 0.15 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method scaling_uniform --experiment_name miscaling_uniform_0.15 --dataset coco --model hidden &
# # CUDA_VISIBLE_DEVICES=2 python certify_wm.py --base_model miweights/COCO/hidden/miscaling_uniform_0.2/epoch_99_state.pth --sigma 0.2 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method scaling_uniform --experiment_name miscaling_uniform_0.2 --dataset coco --model hidden &

# # miAffine
# CUDA_VISIBLE_DEVICES=3 python certify_wm.py --base_model miweights/COCO/hidden/miaffine_0.01/epoch_99_state.pth --sigma 0.01 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name miaffine_0.01 --dataset coco --model hidden &
# CUDA_VISIBLE_DEVICES=4 python certify_wm.py --base_model miweights/COCO/hidden/miaffine_0.02/epoch_99_state.pth --sigma 0.02 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name miaffine_0.02 --dataset coco --model hidden &
# CUDA_VISIBLE_DEVICES=5 python certify_wm.py --base_model miweights/COCO/hidden/miaffine_0.03/epoch_99_state.pth --sigma 0.03 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name miaffine_0.03 --dataset coco --model hidden