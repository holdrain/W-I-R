# hidden celeba

# GN
# CUDA_VISIBLE_DEVICES=1,0 accelerate launch --num_processes 2 --main_process_port 29501 Hidden_train.py --file options/Hidden/mi/Dmi_opt.yml --dataset celeb --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice GN --sigma 0.1 --lr 0.0001 --resume_path weights/CelebA/hidden/OO/epoch_499_state100.pth &
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29502 Hidden_train.py --file options/Hidden/mi/Dmi_opt.yml --dataset celeb --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice GN --sigma 0.25 --lr 0.0001 --resume_path weights/CelebA/hidden/OO/epoch_499_state100.pth &
# wait
# CUDA_VISIBLE_DEVICES=4,5  accelerate launch --num_processes 2 --main_process_port 29503 Hidden_train.py --file options/Hidden/mi/Dmi_opt.yml --dataset celeb --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice GN --sigma 0.5 --lr 0.0001 --resume_path weights/CelebA/hidden/OO/epoch_499_state100.pth & 

# # Affine
# CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num_processes 2 --main_process_port 29504 Hidden_train.py --file options/Hidden/mi/Dmi_opt.yml --dataset celeb --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice affine --sigma 0.01 --lr 0.0001 --resume_path weights/CelebA/hidden/OO/epoch_499_state100.pth
# wait
# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 29505 Hidden_train.py --file options/Hidden/mi/Dmi_opt.yml --dataset celeb --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice affine --sigma 0.02 --lr 0.0001 --resume_path weights/CelebA/hidden/OO/epoch_499_state100.pth &


# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 29506 Hidden_train.py --file options/Hidden/mi/Dmi_opt.yml --dataset celeb --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice affine --sigma 0.03 --lr 0.0001 --resume_path weights/CelebA/hidden/OO/epoch_499_state100.pth &
# # COCO
# # GN
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29507 Hidden_train.py --file options/Hidden/mi/Dmi_opt.yml --dataset coco --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice GN --sigma 0.1 --lr 0.0001 --resume_path weights/COCO/hidden/OO/epoch_499_state100.pth --num 10000 &
# CUDA_VISIBLE_DEVICES=5,4 accelerate launch --num_processes 2 --main_process_port 29508 Hidden_train.py --file options/Hidden/mi/Dmi_opt.yml --dataset coco --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice GN --sigma 0.25 --lr 0.0001 --resume_path weights/COCO/hidden/OO/epoch_499_state100.pth --num 10000 &
# CUDA_VISIBLE_DEVICES=6,7  accelerate launch --num_processes 2 --main_process_port 29509 Hidden_train.py --file options/Hidden/mi/Dmi_opt.yml --dataset coco --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice GN --sigma 0.5 --lr 0.0001 --resume_path weights/COCO/hidden/OO/epoch_499_state100.pth --num 10000
# wait
# # # Affine
# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 29510 Hidden_train.py --file options/Hidden/mi/Dmi_opt.yml --dataset coco --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice affine --sigma 0.01 --lr 0.0001 --resume_path weights/COCO/hidden/OO/epoch_499_state100.pth --num 10000 &
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29511 Hidden_train.py --file options/Hidden/mi/Dmi_opt.yml --dataset coco --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice affine --sigma 0.02 --lr 0.0001 --resume_path weights/COCO/hidden/OO/epoch_499_state100.pth --num 10000 &
# CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num_processes 2 --main_process_port 29512 Hidden_train.py --file options/Hidden/mi/Dmi_opt.yml --dataset coco --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice affine --sigma 0.03 --lr 0.0001 --resume_path weights/COCO/hidden/OO/epoch_499_state100.pth --num 10000





# tuning (8.25)

# CelebA
# CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num_processes 2 --main_process_port 29501 Hidden_train.py --file options/Hidden/mi/Dmi_opt.yml --dataset celeb --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice GN --sigma 0.1 --lr 0.0001 --resume_path weights/CelebA/hidden/OO/epoch_499_state100.pth &
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29502 Hidden_train.py --file options/Hidden/mi/Dmi_opt.yml --dataset celeb --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice GN --sigma 0.25 --lr 0.0001 --resume_path weights/CelebA/hidden/OO/epoch_499_state100.pth &
# CUDA_VISIBLE_DEVICES=4,5  accelerate launch --num_processes 2 --main_process_port 29503 Hidden_train.py --file options/Hidden/mi/Dmi_opt.yml --dataset celeb --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice GN --sigma 0.5 --lr 0.0001 --resume_path weights/CelebA/hidden/OO/epoch_499_state100.pth &


# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 29506 Hidden_train.py --file options/Hidden/mi/Dmi_opt.yml --dataset celeb --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice affine --sigma 0.03 --lr 0.0001 --resume_path weights/CelebA/hidden/OO/epoch_499_state100.pth

# COCO
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29507 Hidden_train.py --file options/Hidden/mi/Dmi_opt.yml --dataset coco --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice GN --sigma 0.1 --lr 0.0001 --resume_path weights/COCO/hidden/OO/epoch_499_state100.pth --num 10000 &
# CUDA_VISIBLE_DEVICES=5,4 accelerate launch --num_processes 2 --main_process_port 29508 Hidden_train.py --file options/Hidden/mi/Dmi_opt.yml --dataset coco --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice GN --sigma 0.25 --lr 0.0001 --resume_path weights/COCO/hidden/OO/epoch_499_state100.pth --num 10000 &
# CUDA_VISIBLE_DEVICES=0,1  accelerate launch --num_processes 2 --main_process_port 29509 Hidden_train.py --file options/Hidden/mi/Dmi_opt.yml --dataset coco --batch_size 96 \
# --set_start_epoch 0 --epoch 100 --noise_choice GN --sigma 0.5 --lr 0.0001 --resume_path weights/COCO/hidden/OO/epoch_499_state100.pth --num 10000


# Emperical training with mi
# /mnt/shared/Huggingface/sharedcode/Stegastamp_Train/weights/CelebA/hidden/emperical/epoch_499_state.pth
# celebA
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 29506 Hidden_train.py --file options/Hidden/mi/Dmi_opt.yml --dataset celeb --batch_size 96 \
--set_start_epoch 0 --epoch 100 --noise_choice hidden_combined --sigma 0 --lr 0.0001 --resume_path /mnt/shared/Huggingface/sharedcode/Stegastamp_Train/weights/CelebA/hidden/emperical/epoch_499_state.pth
CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num_processes 2 --main_process_port 29509 Hidden_train.py --file options/Hidden/mi/Dmi_opt.yml --dataset celeb --batch_size 96 \
--set_start_epoch 0 --epoch 100 --noise_choice hidden_combined --sigma 0 --lr 0.0001 --resume_path /mnt/shared/Huggingface/sharedcode/Stegastamp_Train/weights/CelebA/hidden/emperical/epoch_499_state.pth
# coco
CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num_processes 2 --main_process_port 29508 Hidden_train.py --file options/Hidden/mi/Dmi_opt.yml --dataset coco --batch_size 96 \
--set_start_epoch 0 --epoch 100 --noise_choice hidden_combined --sigma 0 --lr 0.0001 --resume_path /mnt/shared/Huggingface/sharedcode/Stegastamp_Train/experiments_hidden/coco/hidden_combined_0.0/2024-09-02-09:31-train/path_checkpoint/epoch_699_state.pth --num 10000