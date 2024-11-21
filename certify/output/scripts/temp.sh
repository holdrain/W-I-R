# CUDA_VISIBLE_DEVICES=6 python certify_wm.py --base_model miweights/CelebA/hidden/miaffine_0.01/epoch_99_state.pth --sigma 0.03 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name affine_0.03 --dataset celeb --model hidden \
# --start 0 --end 250 &
# CUDA_VISIBLE_DEVICES=7 python certify_wm.py --base_model miweights/CelebA/hidden/miaffine_0.01/epoch_99_state.pth --sigma 0.03 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name affine_0.03 --dataset celeb --model hidden \
# --start 250 --end 500 

# CUDA_VISIBLE_DEVICES=2 python certify_wm.py --base_model weights/COCO/hidden/GN_0.1/epoch_99_state.pth --sigma 0.1 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name GN_0.1 --dataset coco --model hidden \
# --start 0 --end 250


# CUDA_VISIBLE_DEVICES=6 python certify_wm.py --base_model weights/COCO/hidden/GN_0.1/epoch_99_state.pth --sigma 0.25 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name GN_0.25 --dataset coco --model hidden \
# --start 78 --end 250 &
# CUDA_VISIBLE_DEVICES=7 python certify_wm.py --base_model weights/COCO/hidden/GN_0.1/epoch_99_state.pth --sigma 0.25 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name GN_0.25 --dataset coco --model hidden \
# --start 250 --end 500


# CUDA_VISIBLE_DEVICES=0 python certify_wm.py --base_model miweights/COCO/hidden/miaffine_0.01/epoch_99_state.pth --sigma 0.01 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name miaffine_0.01 --dataset coco --model hidden \
# --start 0 --end 250 &
# CUDA_VISIBLE_DEVICES=2 python certify_wm.py --base_model miweights/COCO/hidden/miaffine_0.01/epoch_99_state.pth --sigma 0.01 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name miaffine_0.01 --dataset coco --model hidden \
# --start 250 --end 500

# CUDA_VISIBLE_DEVICES=0 python certify_wm.py --base_model miweights/COCO/hidden/miaffine_0.02/epoch_99_state.pth --sigma 0.02 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name miaffine_0.02 --dataset coco --model hidden \
# --start 0 --end 250 &
# CUDA_VISIBLE_DEVICES=3 python certify_wm.py --base_model miweights/COCO/hidden/miaffine_0.02/epoch_99_state.pth --sigma 0.02 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name miaffine_0.02 --dataset coco --model hidden \
# --start 250 --end 500

# CUDA_VISIBLE_DEVICES=6 python certify_wm.py --base_model miweights/COCO/hidden/miaffine_0.03/epoch_99_state.pth --sigma 0.03 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name miaffine_0.03 --dataset coco --model hidden \
# --start 0 --end 250 &
# CUDA_VISIBLE_DEVICES=7 python certify_wm.py --base_model miweights/COCO/hidden/miaffine_0.03/epoch_99_state.pth --sigma 0.03 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name miaffine_0.03 --dataset coco --model hidden \
# --start 250 --end 500\

# CUDA_VISIBLE_DEVICES=4 python certify_wm.py --base_model weights/CelebA/hidden/affine_0.03/epoch_99_state.pth --sigma 0.03 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name affine_0.03 --dataset celeb --model hidden \
# --start 250 --end 350 &
# CUDA_VISIBLE_DEVICES=5 python certify_wm.py --base_model weights/CelebA/hidden/affine_0.03/epoch_99_state.pth --sigma 0.03 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name affine_0.03 --dataset celeb --model hidden \
# --start 350 --end 450 &
# CUDA_VISIBLE_DEVICES=7 python certify_wm.py --base_model weights/CelebA/hidden/affine_0.03/epoch_99_state.pth --sigma 0.03 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name affine_0.03 --dataset celeb --model hidden \
# --start 450 --end 500


# CUDA_VISIBLE_DEVICES=0 python certify_wm.py --base_model weights/COCO/hidden/affine_0.02/epoch_99_state.pth --sigma 0.02 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name affine_0.02 --dataset coco --model hidden \
# --start 441 --end 460 &
# CUDA_VISIBLE_DEVICES=2 python certify_wm.py --base_model weights/COCO/hidden/affine_0.02/epoch_99_state.pth --sigma 0.02 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name affine_0.02 --dataset coco --model hidden \
# --start 460 --end 480 &
# CUDA_VISIBLE_DEVICES=7 python certify_wm.py --base_model weights/COCO/hidden/affine_0.02/epoch_99_state.pth --sigma 0.02 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method affine --experiment_name affine_0.02 --dataset coco --model hidden \
# --start 480 --end 500

# CUDA_VISIBLE_DEVICES=0 python certify_wm.py --base_model miweights/COCO/hidden/miGN_0.5/epoch_99_state.pth --sigma 0.5 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name miGN_0.5 --dataset coco --model hidden \
# --start 0 --end 15 &
# CUDA_VISIBLE_DEVICES=2 python certify_wm.py --base_model miweights/COCO/hidden/miGN_0.5/epoch_99_state.pth --sigma 0.5 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name miGN_0.5 --dataset coco --model hidden \
# --start 15 --end 30 &
# CUDA_VISIBLE_DEVICES=3 python certify_wm.py --base_model miweights/COCO/hidden/miGN_0.5/epoch_99_state.pth --sigma 0.5 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name miGN_0.5 --dataset coco --model hidden \
# --start 30 --end 55


# CUDA_VISIBLE_DEVICES=6 python certify_wm.py --base_model weights/COCO/hidden/GN_0.1/epoch_99_state.pth --sigma 0.25 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name GN_0.25 --dataset coco --model hidden \
# --start 78 --end 250 &
# CUDA_VISIBLE_DEVICES=7 python certify_wm.py --base_model weights/COCO/hidden/GN_0.1/epoch_99_state.pth --sigma 0.25 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name GN_0.25 --dataset coco --model hidden \
# --start 250 --end 500