CUDA_VISIBLE_DEVICES=3 python certify_wm.py --base_model weights/COCO/hidden/GN_0.25/epoch_99_state.pth --sigma 0.25 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name GN_0.25 --dataset coco --model hidden \
--start 0 --end 250 &
CUDA_VISIBLE_DEVICES=4 python certify_wm.py --base_model weights/COCO/hidden/GN_0.25/epoch_99_state.pth --sigma 0.25 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name GN_0.25 --dataset coco --model hidden \
--start 250 --end 500



# CUDA_VISIBLE_DEVICES=4 python certify_wm.py --base_model weights/COCO/hidden/GN_0.5/epoch_99_state.pth --sigma 0.5 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name GN_0.5 --dataset coco --model hidden \
# --start 0 --end 250 &
# CUDA_VISIBLE_DEVICES=5 python certify_wm.py --base_model weights/COCO/hidden/GN_0.5/epoch_99_state.pth --sigma 0.5 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name GN_0.5 --dataset coco --model hidden \
# --start 250 --end 500

# CUDA_VISIBLE_DEVICES=5 python certify_wm.py --base_model miweights/COCO/hidden/miGN_0.5/epoch_99_state.pth --sigma 0.5 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name miGN_0.5 --dataset coco --model hidden \
# --start 0 --end 250 

# # CUDA_VISIBLE_DEVICES=5 python certify_wm.py --base_model weights/COCO/hidden/GN_0.5/epoch_99_state.pth --sigma 0.5 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name GN_0.5 --dataset coco --model hidden \
# # --start 250 --end 500



# CUDA_VISIBLE_DEVICES=0 python certify_wm.py --base_model weights/COCO/hidden/GN_0.1/epoch_99_state.pth --sigma 0.1 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN  --experiment_name newGN_0.1 --dataset coco --model hidden \
# --start 0 --end 150 &
# CUDA_VISIBLE_DEVICES=1 python certify_wm.py --base_model weights/COCO/hidden/GN_0.1/epoch_99_state.pth --sigma 0.1 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name newGN_0.1 --dataset coco --model hidden \
# --start 150 --end 400 &
# CUDA_VISIBLE_DEVICES=2 python certify_wm.py --base_model weights/COCO/hidden/GN_0.1/epoch_99_state.pth --sigma 0.1 --skip 1 --max 500 --certify_batch_sz 1000 --certify_method GN --experiment_name newGN_0.1 --dataset coco --model hidden \
# --start 400 --end 500


# CUDA_VISIBLE_DEVICES=2 python certify_wm.py --base_model miweights/COCO/stega/miGN_0.1/epoch_99_state.pth --sigma 0.1 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method GN  --experiment_name newmiGN_0.1 --dataset coco --model stega \
# --start 0 --end 250 &
# CUDA_VISIBLE_DEVICES=6 python certify_wm.py --base_model miweights/COCO/stega/miGN_0.1/epoch_99_state.pth --sigma 0.1 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method GN --experiment_name newmiGN_0.1 --dataset coco --model stega \
# --start 250 --end 500 



# CUDA_VISIBLE_DEVICES=5 python certify_wm.py --base_model miweights/COCO/stega/miGN_0.25/epoch_99_state.pth --sigma 0.25 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method GN  --experiment_name newmiGN_0.25 --dataset coco --model stega \
# --start 0 --end 250 &
# CUDA_VISIBLE_DEVICES=7 python certify_wm.py --base_model miweights/COCO/stega/miGN_0.25/epoch_99_state.pth --sigma 0.25 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method GN --experiment_name newmiGN_0.25 --dataset coco --model stega \
# --start 250 --end 500


# CUDA_VISIBLE_DEVICES=2 python certify_wm.py --base_model miweights/COCO/stega/miaffine_0.01/epoch_99_state.pth --sigma 0.01 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method affine  --experiment_name newmiaffine_0.01 --dataset coco --model stega \
# --start 0 --end 250 &
# CUDA_VISIBLE_DEVICES=7 python certify_wm.py --base_model miweights/COCO/stega/miaffine_0.01/epoch_99_state.pth --sigma 0.01 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method affine --experiment_name newmiaffine_0.01 --dataset coco --model stega \
# --start 250 --end 500

# CUDA_VISIBLE_DEVICES=5 python certify_wm.py --base_model miweights/COCO/stega/miaffine_0.02/epoch_99_state.pth --sigma 0.02 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method affine  --experiment_name newmiaffine_0.02 --dataset coco --model stega \
# --start 0 --end 250 &
# CUDA_VISIBLE_DEVICES=6 python certify_wm.py --base_model miweights/COCO/stega/miaffine_0.02/epoch_99_state.pth --sigma 0.02 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method affine --experiment_name newmiaffine_0.02 --dataset coco --model stega \
# --start 250 --end 500

# CUDA_VISIBLE_DEVICES=0 python certify_wm.py --base_model weights/COCO/stega/affine_0.03/epoch_99_state.pth --sigma 0.01 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method affine  --experiment_name new1affine_0.01 --dataset coco --model stega \
# --start 0 --end 150 &
# CUDA_VISIBLE_DEVICES=1 python certify_wm.py --base_model weights/COCO/stega/affine_0.03/epoch_99_state.pth --sigma 0.01 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method affine --experiment_name new1affine_0.01 --dataset coco --model stega \
# --start 150 --end 350 &
# CUDA_VISIBLE_DEVICES=2 python certify_wm.py --base_model weights/COCO/stega/affine_0.03/epoch_99_state.pth --sigma 0.01 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method affine --experiment_name new1affine_0.01 --dataset coco --model stega \
# --start 350 --end 500


CUDA_VISIBLE_DEVICES=3 python certify_wm.py --base_model weights/COCO/stega/affine_0.03/epoch_99_state.pth --sigma 0.03 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method affine  --experiment_name new2affine_0.03 --dataset coco --model stega \
--start 0 --end 150 &
CUDA_VISIBLE_DEVICES=6 python certify_wm.py --base_model weights/COCO/stega/affine_0.03/epoch_99_state.pth --sigma 0.03 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method affine --experiment_name new2affine_0.03 --dataset coco --model stega \
--start 150 --end 350 &
CUDA_VISIBLE_DEVICES=7 python certify_wm.py --base_model weights/COCO/stega/affine_0.03/epoch_99_state.pth --sigma 0.03 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method affine --experiment_name new2affine_0.03 --dataset coco --model stega \
--start 350 --end 500


# CUDA_VISIBLE_DEVICES=5 python certify_wm.py --base_model miweights/COCO/stega/miGN_0.5/epoch_99_state.pth --sigma 0.5 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method GN  --experiment_name newmiGN_0.5 --dataset coco --model stega \
# --start 0 --end 250 &
# CUDA_VISIBLE_DEVICES=7 python certify_wm.py --base_model miweights/COCO/stega/miGN_0.5/epoch_99_state.pth --sigma 0.5 --skip 1 --max 500 --certify_batch_sz 5000 --certify_method GN --experiment_name newmiGN_0.5 --dataset coco --model stega \
# --start 250 --end 500
# CUDA_VISIBLE_DEVICES=2 python certify_wm.py --base_model weights/CelebA/stega/GN_0.5/epoch_99_state.pth --sigma 0.5 --skip 1 --max 500 --certify_batch_sz 10000 --certify_method GN  --experiment_name newGN_0.5 --dataset celeb --model stega \
# --start 0 --end 150 &
# CUDA_VISIBLE_DEVICES=6 python certify_wm.py --base_model weights/CelebA/stega/GN_0.5/epoch_99_state.pth --sigma 0.5 --skip 1 --max 500 --certify_batch_sz 10000 --certify_method GN --experiment_name newGN_0.5 --dataset celeb --model stega \
# --start 150 --end 250 &
# CUDA_VISIBLE_DEVICES=7 python certify_wm.py --base_model weights/CelebA/stega/GN_0.5/epoch_99_state.pth --sigma 0.5 --skip 1 --max 500 --certify_batch_sz 10000 --certify_method GN --experiment_name newGN_0.5 --dataset celeb --model stega \
# --start 250 --end 400 &
# CUDA_VISIBLE_DEVICES=5 python certify_wm.py --base_model weights/CelebA/stega/GN_0.5/epoch_99_state.pth --sigma 0.5 --skip 1 --max 500 --certify_batch_sz 10000 --certify_method GN --experiment_name newGN_0.5 --dataset celeb --model stega \
# --start 400 --end 500