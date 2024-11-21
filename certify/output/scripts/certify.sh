# certify at stega(COCO)

# OO
# CUDA_VISIBLE_DEVICES=0 python certify_wm.py --base_model weights/COCO/stega/OO/epoch_499_state.pth --sigma 0 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method OO --experiment_name OO --dataset coco --model stega

# GN
# CUDA_VISIBLE_DEVICES=0 python certify_wm.py --base_model weights/COCO/stega/GN_0.1/epoch_99_state.pth --sigma 0.1 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method GN --experiment_name GN_0.1 --dataset coco --model stega
# CUDA_VISIBLE_DEVICES=0 python certify_wm.py --base_model weights/COCO/stega/GN_0.25/epoch_99_state.pth --sigma 0.25 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method GN --experiment_name GN_0.25 --dataset coco --model stega
# CUDA_VISIBLE_DEVICES=0 python certify_wm.py --base_model weights/COCO/stega/GN_0.5/epoch_99_state.pth --sigma 0.5 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method GN --experiment_name GN_0.5 --dataset coco --model stega

# Scaling
# CUDA_VISIBLE_DEVICES=4 python certify_wm.py --base_model weights/COCO/stega/scaling_uniform_0.1/epoch_99_state.pth --sigma 0.1 --skip 1 --max 500 --certify_batch_sz 10000 --certify_method scaling_uniform --experiment_name scaling_uniform_0.1 --dataset coco --model stega &
# CUDA_VISIBLE_DEVICES=2 python certify_wm.py --base_model weights/COCO/stega/scaling_uniform_0.15/epoch_99_state.pth --sigma 0.15 --skip 1 --max 500 --certify_batch_sz 10000 --certify_method scaling_uniform --experiment_name scaling_uniform_0.15 --dataset coco --model stega &
# CUDA_VISIBLE_DEVICES=3 python certify_wm.py --base_model weights/COCO/stega/scaling_uniform_0.2/epoch_99_state.pth --sigma 0.2 --skip 1 --max 500 --certify_batch_sz 10000 --certify_method scaling_uniform --experiment_name scaling_uniform_0.2 --dataset coco --model stega &


# Affine
# CUDA_VISIBLE_DEVICES=3 python certify_wm.py --base_model weights/COCO/stega/affine_0.01/epoch_99_state.pth --sigma 0.01 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method affine --experiment_name affine_0.01 --dataset coco --model stega &
# CUDA_VISIBLE_DEVICES=4 python certify_wm.py --base_model weights/COCO/stega/affine_0.02/epoch_99_state.pth --sigma 0.02 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method affine --experiment_name affine_0.02 --dataset coco --model stega &
# CUDA_VISIBLE_DEVICES=5 python certify_wm.py --base_model weights/COCO/stega/affine_0.03/epoch_99_state.pth --sigma 0.03 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method affine --experiment_name affine_0.03 --dataset coco --model stega

# miGN
# CUDA_VISIBLE_DEVICES=0 python certify_wm.py --base_model newmiweights/COCO/stega/miGN_0.1/epoch_99_state.pth --sigma 0.1 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method GN --experiment_name miGN_0.1 --dataset coco --model stega &
# CUDA_VISIBLE_DEVICES=4 python certify_wm.py --base_model newmiweights/COCO/stega/miGN_0.25/epoch_99_state.pth --sigma 0.25 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method GN --experiment_name miGN_0.25 --dataset coco --model stega &
# CUDA_VISIBLE_DEVICES=1 python certify_wm.py --base_model newmiweights/COCO/stega/miGN_0.5/epoch_99_state.pth --sigma 0.5 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method GN --experiment_name miGN_0.5 --dataset coco --model stega

# wait
# # miAffine
# CUDA_VISIBLE_DEVICES=0 python certify_wm.py --base_model newmiweights/COCO/stega/miaffine_0.01/epoch_99_state.pth --sigma 0.01 --skip 1 --max 500 --certify_batch_sz 2000 --certify_method affine --experiment_name miaffine_0.01 --dataset coco --model stega &
# CUDA_VISIBLE_DEVICES=4 python certify_wm.py --base_model newmiweights/COCO/stega/miaffine_0.02/epoch_99_state.pth --sigma 0.02 --skip 1 --max 500 --certify_batch_sz 2000 --certify_method affine --experiment_name miaffine_0.02 --dataset coco --model stega &
# CUDA_VISIBLE_DEVICES=1 python certify_wm.py --base_model newmiweights/COCO/stega/miaffine_0.03/epoch_99_state.pth --sigma 0.03 --skip 1 --max 500 --certify_batch_sz 2000 --certify_method affine --experiment_name miaffine_0.03 --dataset coco --model stega




# certify at (stega CelebA)

# GN
# CUDA_VISIBLE_DEVICES=1 python certify_wm.py --base_model weights/CelebA/stega/GN_0.1/epoch_99_state.pth --sigma 0.1 --skip 1 --max 500 --certify_batch_sz 2500 --certify_method GN --experiment_name GN_0.1 --dataset celeb --model stega &
# CUDA_VISIBLE_DEVICES=2 python certify_wm.py --base_model weights/CelebA/stega/GN_0.25/epoch_99_state.pth --sigma 0.25 --skip 1 --max 500 --certify_batch_sz 2500 --certify_method GN --experiment_name GN_0.25 --dataset celeb --model stega &
# CUDA_VISIBLE_DEVICES=3 python certify_wm.py --base_model weights/CelebA/stega/GN_0.5/epoch_199_state.pth --sigma 0.5 --skip 1 --max 500 --certify_batch_sz 2500 --certify_method GN --experiment_name GN_0.5 --dataset celeb --model stega &

#Scaling
# CUDA_VISIBLE_DEVICES=4 python certify_wm.py --base_model weights/CelebA/stega/scaling_uniform_0.1/epoch_99_state.pth --sigma 0.1 --skip 1 --max 500 --certify_batch_sz 10000 --certify_method scaling_uniform --experiment_name scaling_uniform_0.1 --dataset celeb --model stega &
# CUDA_VISIBLE_DEVICES=5 python certify_wm.py --base_model weights/CelebA/stega/scaling_uniform_0.15/epoch_99_state.pth --sigma 0.15 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method scaling_uniform --experiment_name scaling_uniform_0.15 --dataset celeb --model stega &
# CUDA_VISIBLE_DEVICES=6 python certify_wm.py --base_model weights/CelebA/stega/scaling_uniform_0.2/epoch_99_state.pth --sigma 0.2 --skip 1 --max 500 --certify_batch_sz 2000 --certify_method scaling_uniform --experiment_name scaling_uniform_0.2 --dataset celeb --model stega


# Affine
# CUDA_VISIBLE_DEVICES=4 python certify_wm.py --base_model weights/CelebA/stega/affine_0.01/epoch_99_state.pth --sigma 0.01 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name affine_0.01 --dataset celeb --model stega &
# CUDA_VISIBLE_DEVICES=2 python certify_wm.py --base_model weights/CelebA/stega/affine_0.02/epoch_99_state.pth --sigma 0.02 --skip 1 --max 500 --certify_batch_sz 10000 --certify_method affine --experiment_name affine_0.02 --dataset celeb --model stega &
# CUDA_VISIBLE_DEVICES=3 python certify_wm.py --base_model weights/CelebA/stega/affine_0.03/epoch_99_state.pth --sigma 0.03 --skip 1 --max 500 --certify_batch_sz 10000 --certify_method affine --experiment_name affine_0.03 --dataset celeb --model stega &
# 
# wait
# miGN
# CUDA_VISIBLE_DEVICES=4 python certify_wm.py --base_model newmiweights/CelebA/stega/miGN_0.1/epoch_99_state.pth --sigma 0.1 --skip 1 --max 500 --certify_batch_sz 10000 --certify_method GN --experiment_name miGN_0.1 --dataset celeb --model stega &
# CUDA_VISIBLE_DEVICES=0 python certify_wm.py --base_model newmiweights/CelebA/stega/miGN_0.25/epoch_99_state.pth --sigma 0.25 --skip 1 --max 500 --certify_batch_sz 10000 --certify_method GN --experiment_name miGN_0.25 --dataset celeb --model stega &
# CUDA_VISIBLE_DEVICES=1 python certify_wm.py --base_model newmiweights/CelebA/stega/miGN_0.5/epoch_99_state.pth --sigma 0.5 --skip 1 --max 500 --certify_batch_sz 10000 --certify_method GN --experiment_name miGN_0.5 --dataset celeb --model stega


# # miAffine
# CUDA_VISIBLE_DEVICES=4 python certify_wm.py --base_model newmiweights/CelebA/stega/miaffine_0.01/epoch_99_state.pth --sigma 0.01 --skip 1 --max 500 --certify_batch_sz 8000 --certify_method affine --experiment_name miaffine_0.01 --dataset celeb --model stega &
# CUDA_VISIBLE_DEVICES=1 python certify_wm.py --base_model newmiweights/CelebA/stega/miaffine_0.02/epoch_99_state.pth --sigma 0.02 --skip 1 --max 500 --certify_batch_sz 8000 --certify_method affine --experiment_name miaffine_0.02 --dataset celeb --model stega &
# CUDA_VISIBLE_DEVICES=0 python certify_wm.py --base_model newmiweights/CelebA/stega/miaffine_0.03/epoch_99_state.pth --sigma 0.03 --skip 1 --max 500 --certify_batch_sz 8000 --certify_method affine --experiment_name miaffine_0.03 --dataset celeb --model stega



# certify at (hidden CelebA)

# GN
# CUDA_VISIBLE_DEVICES=1 python certify_wm.py --base_model weights/CelebA/hidden/GN_0.1/epoch_99_state.pth --sigma 0.1 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name GN_0.1 --dataset celeb --model hidden &
# CUDA_VISIBLE_DEVICES=2 python certify_wm.py --base_model weights/CelebA/hidden/GN_0.25/epoch_99_state.pth --sigma 0.25 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name GN_0.25 --dataset celeb --model hidden 
# CUDA_VISIBLE_DEVICES=0 python certify_wm.py --base_model weights/CelebA/hidden/GN_0.5/epoch_99_state.pth --sigma 0.5 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name GN_0.5 --dataset celeb --model hidden &

# # Scaling
# CUDA_VISIBLE_DEVICES=3 python certify_wm.py --base_model weights/CelebA/hidden/scaling_uniform_0.1/epoch_99_state.pth --sigma 0.1 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method scaling_uniform --experiment_name scaling_uniform_0.1 --dataset celeb --model hidden &
# CUDA_VISIBLE_DEVICES=4 python certify_wm.py --base_model weights/CelebA/hidden/scaling_uniform_0.15/epoch_99_state.pth --sigma 0.15 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method scaling_uniform --experiment_name scaling_uniform_0.15 --dataset celeb --model hidden &
# CUDA_VISIBLE_DEVICES=5 python certify_wm.py --base_model weights/CelebA/hidden/scaling_uniform_0.2/epoch_99_state.pth --sigma 0.2 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method scaling_uniform --experiment_name scaling_uniform_0.2 --dataset celeb --model hidden &


# # Affine
# CUDA_VISIBLE_DEVICES=6 python certify_wm.py --base_model weights/CelebA/hidden/affine_0.01/epoch_99_state.pth --sigma 0.01 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name affine_0.01 --dataset celeb --model hidden &
# CUDA_VISIBLE_DEVICES=7 python certify_wm.py --base_model weights/CelebA/hidden/affine_0.02/epoch_99_state.pth --sigma 0.02 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name affine_0.02 --dataset celeb --model hidden
# CUDA_VISIBLE_DEVICES=0 python certify_wm.py --base_model weights/CelebA/hidden/affine_0.03/epoch_99_state.pth --sigma 0.03 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name affine_0.03 --dataset celeb --model hidden &

# miGN
# CUDA_VISIBLE_DEVICES=0 python certify_wm.py --base_model newmiweights/CelebA/hidden/miGN_0.1/epoch_99_state.pth --sigma 0.1 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name miGN_0.1 --dataset celeb --model hidden &
# CUDA_VISIBLE_DEVICES=4 python certify_wm.py --base_model newmiweights/CelebA/hidden/miGN_0.25/epoch_99_state.pth --sigma 0.25 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name miGN_0.25 --dataset celeb --model hidden &
# CUDA_VISIBLE_DEVICES=1 python certify_wm.py --base_model newmiweights/CelebA/hidden/miGN_0.5/epoch_99_state.pth --sigma 0.5 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name miGN_0.5 --dataset celeb --model hidden &


# # miAffine
# CUDA_VISIBLE_DEVICES=0 python certify_wm.py --base_model newmiweights/CelebA/hidden/miaffine_0.01/epoch_99_state.pth --sigma 0.01 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name miaffine_0.01 --dataset celeb --model hidden &
# CUDA_VISIBLE_DEVICES=1 python certify_wm.py --base_model newmiweights/CelebA/hidden/miaffine_0.02/epoch_99_state.pth --sigma 0.02 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name miaffine_0.02 --dataset celeb --model hidden &
# CUDA_VISIBLE_DEVICES=4 python certify_wm.py --base_model newmiweights/CelebA/hidden/miaffine_0.03/epoch_99_state.pth --sigma 0.03 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name miaffine_0.03 --dataset celeb --model hidden 

# # certify at (hidden COCO)
# wait
# # miGN
# CUDA_VISIBLE_DEVICES=1 python certify_wm.py --base_model newmiweights/COCO/hidden/miGN_0.1/epoch_99_state.pth --sigma 0.1 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name miGN_0.1 --dataset coco --model hidden &
# CUDA_VISIBLE_DEVICES=4 python certify_wm.py --base_model newmiweights/COCO/hidden/miGN_0.25/epoch_99_state.pth --sigma 0.25 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name miGN_0.25 --dataset coco --model hidden &
# CUDA_VISIBLE_DEVICES=0 python certify_wm.py --base_model newmiweights/COCO/hidden/miGN_0.5/epoch_99_state.pth --sigma 0.5 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name miGN_0.5 --dataset coco --model hidden 

# wait

# # miAffine
# CUDA_VISIBLE_DEVICES=0 python certify_wm.py --base_model newmiweights/COCO/hidden/miaffine_0.01/epoch_99_state.pth --sigma 0.01 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name miaffine_0.01 --dataset coco --model hidden &
# CUDA_VISIBLE_DEVICES=1 python certify_wm.py --base_model newmiweights/COCO/hidden/miaffine_0.02/epoch_99_state.pth --sigma 0.02 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name miaffine_0.02 --dataset coco --model hidden &
# CUDA_VISIBLE_DEVICES=4 python certify_wm.py --base_model newmiweights/COCO/hidden/miaffine_0.03/epoch_99_state.pth --sigma 0.03 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name miaffine_0.03 --dataset coco --model hidden 


# --------------------------AE--------------------------
# miGN
# CUDA_VISIBLE_DEVICES=1 python certify_wm.py --base_model newmiweights/CelebA/stega/miGN_0.1/epoch_99_state.pth --sigma 0.1 --skip 1 --max 100 --certify_batch_sz 500 --N 10000 --certify_method GN --experiment_name miGN_0.1 --dataset ae --model stega &
# CUDA_VISIBLE_DEVICES=6 python certify_wm.py --base_model newmiweights/CelebA/stega/miGN_0.25/epoch_99_state.pth --sigma 0.25 --skip 1 --max 100 --certify_batch_sz 500 --N 10000 --certify_method GN --experiment_name miGN_0.25 --dataset ae --model stega &
# CUDA_VISIBLE_DEVICES=3 python certify_wm.py --base_model newmiweights/CelebA/stega/miGN_0.5/epoch_99_state.pth --sigma 0.5 --skip 1 --max 100 --certify_batch_sz 500 --N 10000 --certify_method GN --experiment_name miGN_0.5 --dataset ae --model stega

# miAffine
CUDA_VISIBLE_DEVICES=1 python certify_wm.py --base_model newmiweights/CelebA/stega/miaffine_0.01/epoch_99_state.pth --sigma 0.01 --skip 1 --max 500 --certify_batch_sz 500 --N 10000 --certify_method affine --experiment_name miaffine_0.01 --dataset ae --model stega &
CUDA_VISIBLE_DEVICES=6 python certify_wm.py --base_model newmiweights/CelebA/stega/miaffine_0.02/epoch_99_state.pth --sigma 0.02 --skip 1 --max 500 --certify_batch_sz 500 --N 10000 --certify_method affine --experiment_name miaffine_0.02 --dataset ae --model stega &
CUDA_VISIBLE_DEVICES=3 python certify_wm.py --base_model newmiweights/CelebA/stega/miaffine_0.03/epoch_99_state.pth --sigma 0.03 --skip 1 --max 500 --certify_batch_sz 500 --N 10000 --certify_method affine --experiment_name miaffine_0.03 --dataset ae --model stega

