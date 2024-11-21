# hidden celeba
# OO
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch --num_processes 6 --main_process_port 29506 Hidden_train.py --file options/Hidden/oo_opt.yml --dataset celeb --batch_size 96 \
# --set_start_epoch 0 --epoch 500 --noise_choice OO --lr 0.002

# GN
# CUDA_VISIBLE_DEVICES=3,4 accelerate launch --num_processes 2 --main_process_port 29506 Hidden_train.py --file options/Hidden/D_opt.yml --dataset celeb --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice GN --sigma 0.1 --lr 0.0001 --resume_path weights/CelebA/hidden/OO/epoch_499_state100.pth &
# CUDA_VISIBLE_DEVICES=5,6 accelerate launch --num_processes 2 --main_process_port 29507 Hidden_train.py --file options/Hidden/D_opt.yml --dataset celeb --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice GN --sigma 0.25 --lr 0.0001 --resume_path weights/CelebA/hidden/OO/epoch_499_state100.pth &
# CUDA_VISIBLE_DEVICES=7,0 accelerate launch --num_processes 2 --main_process_port 29508 Hidden_train.py --file options/Hidden/D_opt.yml --dataset celeb --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice GN --sigma 0.5 --lr 0.0001 --resume_path weights/CelebA/hidden/OO/epoch_499_state100.pth

# # Scaling
# CUDA_VISIBLE_DEVICES=3,4 accelerate launch --num_processes 2 --main_process_port 29506 Hidden_train.py --file options/Hidden/D_opt.yml --dataset celeb --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice scaling_uniform --sigma 0.1 --lr 0.0001 --resume_path weights/CelebA/hidden/OO/epoch_499_state100.pth &
# CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes 2 --main_process_port 29507 Hidden_train.py --file options/Hidden/D_opt.yml --dataset celeb --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice scaling_uniform --sigma 0.15 --lr 0.0001 --resume_path weights/CelebA/hidden/OO/epoch_499_state100.pth &
# CUDA_VISIBLE_DEVICES=7,0 accelerate launch --num_processes 2 --main_process_port 29508 Hidden_train.py --file options/Hidden/D_opt.yml --dataset celeb --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice scaling_uniform --sigma 0.2 --lr 0.0001 --resume_path weights/CelebA/hidden/OO/epoch_499_state100.pth 

# # Affine
# CUDA_VISIBLE_DEVICES=3,4 accelerate launch --num_processes 2 --main_process_port 29506 Hidden_train.py --file options/Hidden/D_opt.yml --dataset celeb --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice affine --sigma 0.01 --lr 0.0001 --resume_path weights/CelebA/hidden/OO/epoch_499_state100.pth &
# CUDA_VISIBLE_DEVICES=5,6 accelerate launch --num_processes 2 --main_process_port 29507 Hidden_train.py --file options/Hidden/D_opt.yml --dataset celeb --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice affine --sigma 0.02 --lr 0.0001 --resume_path weights/CelebA/hidden/OO/epoch_499_state100.pth &
# CUDA_VISIBLE_DEVICES=7,0 accelerate launch --num_processes 2 --main_process_port 29508 Hidden_train.py --file options/Hidden/D_opt.yml --dataset celeb --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice affine --sigma 0.03 --lr 0.0001 --resume_path weights/CelebA/hidden/OO/epoch_499_state100.pth

# COCO
# OO
# CUDA_VISIBLE_DEVICES=1,2,3,4,5, accelerate launch --num_processes 5 --main_process_port 29501 Hidden_train.py --file options/Hidden/oo_opt.yml --dataset coco --batch_size 96 \
# --set_start_epoch 0 --epoch 500 --noise_choice OO --lr 0.0005 --num 10000


# # GN
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29506 Hidden_train.py --file options/Hidden/D_opt.yml --dataset coco --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice GN --sigma 0.1 --lr 0.0001 --resume_path weights/COCO/hidden/OO/epoch_499_state100.pth  --num 10000 &
# CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num_processes 2 --main_process_port 29507 Hidden_train.py --file options/Hidden/D_opt.yml --dataset coco --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice GN --sigma 0.25 --lr 0.0001 --resume_path weights/COCO/hidden/OO/epoch_499_state100.pth --num 10000&
# CUDA_VISIBLE_DEVICES=7,6 accelerate launch --num_processes 2 --main_process_port 29508 Hidden_train.py --file options/Hidden/D_opt.yml --dataset coco --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice GN --sigma 0.5 --lr 0.0001 --resume_path weights/COCO/hidden/OO/epoch_499_state100.pth --num 10000

# # Scaling
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29506 Hidden_train.py --file options/Hidden/D_opt.yml --dataset coco --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice scaling_uniform --sigma 0.1 --lr 0.0001 --resume_path weights/COCO/hidden/OO/epoch_499_state100.pth --num 10000 &
# CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num_processes 2 --main_process_port 29507 Hidden_train.py --file options/Hidden/D_opt.yml --dataset coco --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice scaling_uniform --sigma 0.15 --lr 0.0001 --resume_path weights/COCO/hidden/OO/epoch_499_state100.pth --num 10000 &
# CUDA_VISIBLE_DEVICES=7,6 accelerate launch --num_processes 2 --main_process_port 29508 Hidden_train.py --file options/Hidden/D_opt.yml --dataset coco --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice scaling_uniform --sigma 0.2 --lr 0.0001 --resume_path weights/COCO/hidden/OO/epoch_499_state100.pth --num 10000

# Affine
# CUDA_VISIBLE_DEVICES=3,2 accelerate launch --num_processes 2 --main_process_port 29506 Hidden_train.py --file options/Hidden/D_opt.yml --dataset coco --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice affine --sigma 0.01 --lr 0.0001 --resume_path weights/COCO/hidden/OO/epoch_499_state100.pth --num 10000 &
# CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num_processes 2 --main_process_port 29507 Hidden_train.py --file options/Hidden/D_opt.yml --dataset coco --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice affine --sigma 0.02 --lr 0.0001 --resume_path weights/COCO/hidden/OO/epoch_499_state100.pth --num 10000 &
# CUDA_VISIBLE_DEVICES=7,6 accelerate launch --num_processes 2 --main_process_port 29508 Hidden_train.py --file options/Hidden/D_opt.yml --dataset coco --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice affine --sigma 0.03 --lr 0.0001 --resume_path weights/COCO/hidden/OO/epoch_499_state100.pth --num 10000



# combined training
CUDA_VISIBLE_DEVICES=7,6 accelerate launch --num_processes 2 --main_process_port 29508 Hidden_train.py --file options/Hidden/D_opt.yml --dataset coco --batch_size 96 \
--set_start_epoch 0 --epoch 500 --noise_choice hidden_combined --sigma 0 --lr 0.0001 --num 10000
CUDA_VISIBLE_DEVICES=1,2 accelerate launch --num_processes 2 --main_process_port 29509 Hidden_train.py --file options/Hidden/D_opt.yml --dataset celeb --batch_size 96 \
--set_start_epoch 0 --epoch 500 --noise_choice hidden_combined --sigma 0 --lr 0.0001

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 29502 Hidden_train.py --file options/Hidden/D_opt.yml --dataset coco --batch_size 96 \
--set_start_epoch 0 --epoch 500 --noise_choice hidden_combined --sigma 0 --lr 0.0002 --num 10000


CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num_processes 2 --main_process_port 29509 Hidden_train.py --file options/Hidden/D_opt.yml --dataset celeb --batch_size 96 \
--set_start_epoch 0 --epoch 500 --noise_choice hidden_combined --sigma 0 --lr 0.0001

CUDA_VISIBLE_DEVICES=2,4 accelerate launch --num_processes 2 --main_process_port 29502 Hidden_train.py --file options/Hidden/D_opt.yml --dataset coco --batch_size 96 \
--set_start_epoch 500 --epoch 700 --noise_choice hidden_combined --sigma 0 --lr 0.0001 --num 10000 --resume_path /mnt/shared/Huggingface/sharedcode/Stegastamp_Train/experiments_hidden/coco/hidden_combined_0.0/2024-09-01-20:32-train/path_checkpoint/epoch_499_state.pth


CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num_processes 2 --main_process_port 29509 Hidden_train.py --file options/Hidden/D_opt.yml --dataset celeb --batch_size 96 \
--set_start_epoch 0 --epoch 500 --noise_choice hidden_combined --sigma 0 --lr 0.0001

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 29502 Hidden_train.py --file options/Hidden/D_opt.yml --dataset coco --batch_size 96 \
--set_start_epoch 0 --epoch 500 --noise_choice hidden_combined --sigma 0 --lr 0.0002 --num 10000


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --num_processes 8 --main_process_port 29509 Hidden_train.py --file options/Hidden/D_opt.yml --dataset celeb --batch_size 120 \
--set_start_epoch 0 --epoch 700 --noise_choice hidden_combined --sigma 0 --lr 0.0002