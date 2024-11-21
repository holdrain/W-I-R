# --------------CelebA--------------

# OO
# CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 accelerate launch --num_processes 6 --main_process_port 29520 Stega_train.py --file options/Stega/OO_opt.yml --dataset celeb --batch_size 500 --noise_choice OO --lre 0.0001 --lrd 0.0001 \
# --epoch 500 --set_start_epoch 0

# GN 
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29501 Stega_train.py --file options/Stega/D_opt.yml --dataset celeb --batch_size 500 --sigma 0.1 --noise_choice GN --resume_path weights/CelebA/stega/OO/epoch_499_state.pth --lre 0.0001 --lrd 0.0001 \
# --epoch 100 &
# CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num_processes 2 --main_process_port 29502 Stega_train.py --file options/Stega/D_opt.yml --dataset celeb --batch_size 500 --sigma 0.25 --noise_choice GN --resume_path weights/CelebA/stega/OO/epoch_499_state.pth --lre 0.0001 --lrd 0.0001 \
# --epoch 100 &
# CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num_processes 2 --main_process_port 29503 Stega_train.py --file options/Stega/D_opt.yml --dataset celeb --batch_size 500 --sigma 0.5 --noise_choice GN --resume_path weights/CelebA/stega/OO/epoch_499_state.pth --lre 0.0001 --lrd 0.0001 \
# --epoch 100 

# # affine
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29501 Stega_train.py --file options/Stega/D_opt.yml --dataset celeb --batch_size 480 --sigma 0.01 --noise_choice affine --resume_path weights/CelebA/stega/OO/epoch_499_state.pth --lre 0.0001 --lrd 0.0001 \
# --epoch 100 &
# CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num_processes 2 --main_process_port 29502 Stega_train.py --file options/Stega/D_opt.yml --dataset celeb --batch_size 480 --sigma 0.02 --noise_choice affine --resume_path weights/CelebA/stega/OO/epoch_499_state.pth --lre 0.0001 --lrd 0.0001 \
# --epoch 100 &
# CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num_processes 2 --main_process_port 29503 Stega_train.py --file options/Stega/D_opt.yml --dataset celeb --batch_size 480 --sigma 0.03 --noise_choice affine --resume_path weights/CelebA/stega/OO/epoch_499_state.pth --lre 0.0001 --lrd 0.0001 \
# --epoch 100 

# # scaling
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29501 Stega_train.py --file options/Stega/D_opt.yml --dataset celeb --batch_size 480 --sigma 0.1 --noise_choice scaling_uniform --resume_path weights/CelebA/stega/OO/epoch_499_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --optimizer adam &
# CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num_processes 2 --main_process_port 29502 Stega_train.py --file options/Stega/D_opt.yml --dataset celeb --batch_size 480 --sigma 0.15 --noise_choice scaling_uniform --resume_path weights/CelebA/stega/OO/epoch_499_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --optimizer adam 
# CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num_processes 2 --main_process_port 29503 Stega_train.py --file options/Stega/D_opt.yml --dataset celeb --batch_size 480 --sigma 0.2 --noise_choice scaling_uniform --resume_path weights/CelebA/stega/OO/epoch_499_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --optimizer adam 

# additional trainning
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29501 Stega_train.py --file options/Stega/D_opt.yml --dataset celeb --batch_size 480 --sigma 0.5 --noise_choice GN --resume_path experiments_stegastamp/celeb/GN_0.5/-2024-05-29-22:43-train/path_checkpoint/epoch_99_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 100 --epoch 200 &
# CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num_processes 2 --main_process_port 29502 Stega_train.py --file options/Stega/D_opt.yml --dataset celeb --batch_size 480 --sigma 0.03 --noise_choice affine --resume_path experiments_stegastamp/celeb/affine_0.03/-2024-05-30-02:01-train/path_checkpoint/epoch_99_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 100 --epoch 200 &
# CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num_processes 2 --main_process_port 29503 Stega_train.py --file options/Stega/D_opt.yml --dataset celeb --batch_size 480 --sigma 0.2 --noise_choice scaling_uniform --resume_path experiments_stegastamp/celeb/scaling_uniform_0.2/-2024-05-30-05:20-train/path_checkpoint/epoch_99_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 100 --epoch 200

# new kl loss














# --------------COCO--------------
# OO
# CUDA_VISIBLE_DEVICES=0,2,4,5 accelerate launch --num_processes 4 --main_process_port 29520 Stega_train.py --file options/Stega/OO_opt.yml --dataset imagenet --batch_size 128 --noise_choice OO \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 500 --optimizer adam --num 20000

# # GN
# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 29520 Stega_train.py --file options/Stega/D_opt.yml --dataset coco --batch_size 500 --sigma 0.1 --noise_choice GN --resume_path weights/COCO/stega/OO/epoch_499_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --optimizer adam --num 10000 &
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29521 Stega_train.py --file options/Stega/D_opt.yml --dataset coco --batch_size 500 --sigma 0.25 --noise_choice GN --resume_path weights/COCO/stega/OO/epoch_499_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --optimizer adam --num 10000 &
# CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num_processes 2 --main_process_port 29522 Stega_train.py --file options/Stega/D_opt.yml --dataset coco --batch_size 500 --sigma 0.5 --noise_choice GN --resume_path weights/COCO/stega/OO/epoch_499_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --optimizer adam --num 10000

# affine
# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 29520 Stega_train.py --file options/Stega/D_opt.yml --dataset coco --batch_size 500 --sigma 0.01 --noise_choice affine --resume_path weights/COCO/stega/OO/epoch_499_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --optimizer adam --num 10000 &
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29521 Stega_train.py --file options/Stega/D_opt.yml --dataset coco --batch_size 500 --sigma 0.02 --noise_choice affine --resume_path weights/COCO/stega/OO/epoch_499_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --optimizer adam --num 10000 &
# CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num_processes 2 --main_process_port 29522 Stega_train.py --file options/Stega/D_opt.yml --dataset coco --batch_size 500 --sigma 0.03 --noise_choice affine --resume_path weights/COCO/stega/OO/epoch_499_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --optimizer adam --num 10000

# sleep 18000

# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 29520 Stega_train.py --file options/Stega/D_opt.yml --dataset coco --batch_size 500 --sigma 0.1 --noise_choice scaling_uniform --resume_path weights/COCO/stega/OO/epoch_499_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --optimizer adam --num 10000 &
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29521 Stega_train.py --file options/Stega/D_opt.yml --dataset coco --batch_size 500 --sigma 0.15 --noise_choice scaling_uniform --resume_path weights/COCO/stega/OO/epoch_499_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --optimizer adam --num 10000 &
# CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num_processes 2 --main_process_port 29522 Stega_train.py --file options/Stega/D_opt.yml --dataset coco --batch_size 500 --sigma 0.2 --noise_choice scaling_uniform --resume_path weights/COCO/stega/OO/epoch_499_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --optimizer adam --num 10000


# CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num_processes 2 --main_process_port 29522 Stega_train.py --file options/Stega/D_opt.yml --dataset coco --batch_size 500 --sigma 0.1 --noise_choice combined --resume_path weights/COCO/stega/OO/epoch_499_state100.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --optimizer adam --num 10000



# combined training
# CUDA_VISIBLE_DEVICES=0,2,4,5 accelerate launch --num_processes 4 --main_process_port 29521 Stega_train.py --file options/Stega/D_opt.yml --dataset coco --batch_size 128 --sigma 0.1 --noise_choice GN \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --optimizer adam --num 20000 --resume_path adam /data/shared/Huggingface/sharedcode/Stegastamp_Train/experiments_stegastamp/imagenet/OO/-2024-11-20-11:31-train/path_checkpoint/epoch_499_state.pth


# ------------IMAGENET----------------
CUDA_VISIBLE_DEVICES=0,2,4,5 accelerate launch --num_processes 4 --main_process_port 29521 Stega_train.py --file options/Stega/D_opt.yml --dataset imagenet --batch_size 120 --sigma 0.1 --noise_choice GN \
--lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --optimizer adam --num 20000 --resume_path /data/shared/Huggingface/sharedcode/Stegastamp_Train/experiments_stegastamp/imagenet/OO/-2024-11-20-11:31-train/path_checkpoint/epoch_499_state.pth
CUDA_VISIBLE_DEVICES=0,2,4,5 accelerate launch --num_processes 4 --main_process_port 29521 Stega_train.py --file options/Stega/D_opt.yml --dataset imagenet --batch_size 120 --sigma 0.25 --noise_choice GN \
--lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --optimizer adam --num 20000 --resume_path /data/shared/Huggingface/sharedcode/Stegastamp_Train/experiments_stegastamp/imagenet/OO/-2024-11-20-11:31-train/path_checkpoint/epoch_499_state.pth
CUDA_VISIBLE_DEVICES=0,2,4,5 accelerate launch --num_processes 4 --main_process_port 29521 Stega_train.py --file options/Stega/D_opt.yml --dataset imagenet --batch_size 120 --sigma 0.5 --noise_choice GN \
--lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --optimizer adam --num 20000 --resume_path /data/shared/Huggingface/sharedcode/Stegastamp_Train/experiments_stegastamp/imagenet/OO/-2024-11-20-11:31-train/path_checkpoint/epoch_499_state.pth
CUDA_VISIBLE_DEVICES=0,2,4,5 accelerate launch --num_processes 4 --main_process_port 29521 Stega_train.py --file options/Stega/D_opt.yml --dataset imagenet --batch_size 120 --sigma 0.01 --noise_choice affine \
--lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --optimizer adam --num 20000 --resume_path /data/shared/Huggingface/sharedcode/Stegastamp_Train/experiments_stegastamp/imagenet/OO/-2024-11-20-11:31-train/path_checkpoint/epoch_499_state.pth
CUDA_VISIBLE_DEVICES=0,2,4,5 accelerate launch --num_processes 4 --main_process_port 29521 Stega_train.py --file options/Stega/D_opt.yml --dataset imagenet --batch_size 120 --sigma 0.02 --noise_choice affine \
--lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --optimizer adam --num 20000 --resume_path /data/shared/Huggingface/sharedcode/Stegastamp_Train/experiments_stegastamp/imagenet/OO/-2024-11-20-11:31-train/path_checkpoint/epoch_499_state.pth
CUDA_VISIBLE_DEVICES=0,2,4,5 accelerate launch --num_processes 4 --main_process_port 29521 Stega_train.py --file options/Stega/D_opt.yml --dataset imagenet --batch_size 120 --sigma 0.03 --noise_choice affine \
--lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --optimizer adam --num 20000 --resume_path /data/shared/Huggingface/sharedcode/Stegastamp_Train/experiments_stegastamp/imagenet/OO/-2024-11-20-11:31-train/path_checkpoint/epoch_499_state.pth
