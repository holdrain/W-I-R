# ------------celeba----------

# GN
# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 29514 Stega_train.py --file options/Stega/mi/Dmi_opt.yml --dataset celeb --batch_size 480 --sigma 0.1 --noise_choice GN --resume_path experiments_stegastamp/celeb/OO/-2024-05-28-20:11-train/path_checkpoint/epoch_499_state.pth --lre 0.0001 --lrd 0.0001 \
# --epoch 100 --optimizer adam &
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29512 Stega_train.py --file options/Stega/mi/Dmi_opt.yml --dataset celeb --batch_size 500 --sigma 0.25 --noise_choice GN --resume_path experiments_stegastamp/celeb/OO/-2024-05-28-20:11-train/path_checkpoint/epoch_499_state.pth --lre 0.0001 --lrd 0.0001 \
# --epoch 100 --optimizer adam &
# CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num_processes 2 --main_process_port 29513 Stega_train.py --file options/Stega/mi/Dmi_opt.yml --dataset celeb --batch_size 500 --sigma 0.5 --noise_choice GN --resume_path experiments_stegastamp/celeb/OO/-2024-05-28-20:11-train/path_checkpoint/epoch_499_state.pth --lre 0.0001 --lrd 0.0001 \
# --epoch 100 --optimizer adam &
# # CUDA_VISIBLE_DEVICES=2,4 accelerate launch --num_processes 2 --main_process_port 29512 Stega_train.py --file options/Stega/mi/Dmi_opt.yml --dataset celeb --batch_size 480 --sigma 0.01 --noise_choice affine --resume_path experiments_stegastamp/celeb/OO/-2024-05-28-20:11-train/path_checkpoint/epoch_499_state.pth \
# # --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100
# # wait
# # CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 29513 Stega_train.py --file options/Stega/mi/Dmi_opt.yml --dataset celeb --batch_size 480 --sigma 0.02 --noise_choice affine --resume_path experiments_stegastamp/celeb/OO/-2024-05-28-20:11-train/path_checkpoint/epoch_499_state.pth \
# # --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 &
# CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num_processes 2 --main_process_port 29515 Stega_train.py --file options/Stega/mi/Dmi_opt.yml --dataset celeb --batch_size 480 --sigma 0.03 --noise_choice affine --resume_path experiments_stegastamp/celeb/OO/-2024-05-28-20:11-train/path_checkpoint/epoch_499_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 



# -------------COCO------------
# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 29513 Stega_train.py --file options/Stega/mi/Dmi_opt.yml --dataset coco --batch_size 500 --sigma 0.1 --noise_choice GN --resume_path experiments_stegastamp/coco/OO/-2024-05-27-11:49-train/path_checkpoint/epoch_499_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --num 10000 &
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29514 Stega_train.py --file options/Stega/mi/Dmi_opt.yml --dataset coco --batch_size 500 --sigma 0.25 --noise_choice GN --resume_path experiments_stegastamp/coco/OO/-2024-05-27-11:49-train/path_checkpoint/epoch_499_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --num 10000 &
# CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num_processes 2 --main_process_port 29515 Stega_train.py --file options/Stega/mi/Dmi_opt.yml --dataset coco --batch_size 500 --sigma 0.5 --noise_choice GN --resume_path experiments_stegastamp/coco/OO/-2024-05-27-11:49-train/path_checkpoint/epoch_499_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --num 10000

# # affine
# # CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 29513 Stega_train.py --file options/Stega/mi/Dmi_opt.yml --dataset coco --batch_size 500 --sigma 0.01 --noise_choice affine --resume_path weights/COCO/stega/OO/epoch_499_state.pth \
# # --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 200 --num 10000 &
# # CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 29514 Stega_train.py --file options/Stega/mi/Dmi_opt.yml --dataset coco --batch_size 500 --sigma 0.02 --noise_choice affine --resume_path weights/COCO/stega/OO/epoch_499_state.pth \
# # --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 200 --num 10000 &
# CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num_processes 2 --main_process_port 29516 Stega_train.py --file options/Stega/mi/Dmi_opt.yml --dataset coco --batch_size 500 --sigma 0.03 --noise_choice affine --resume_path experiments_stegastamp/coco/OO/-2024-05-27-11:49-train/path_checkpoint/epoch_499_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 200 --num 10000





# 8.26
# celeba
# CUDA_VISIBLE_DEVICES=6,7 accelerate launch --num_processes 2 --main_process_port 29512 Stega_train.py --file options/Stega/mi/Dmi_opt.yml --dataset celeb --batch_size 480 --sigma 0.03 --noise_choice affine --resume_path experiments_stegastamp/celeb/OO/-2024-05-28-20:11-train/path_checkpoint/epoch_499_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 200 &

# # coco
# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 29513 Stega_train.py --file options/Stega/mi/Dmi_opt.yml --dataset coco --batch_size 500 --sigma 0.1 --noise_choice GN --resume_path experiments_stegastamp/coco/OO/-2024-05-27-11:49-train/path_checkpoint/epoch_499_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 200 --num 10000 &
# CUDA_VISIBLE_DEVICES=2,3 accelerate launch --num_processes 2 --main_process_port 29514 Stega_train.py --file options/Stega/mi/Dmi_opt.yml --dataset coco --batch_size 500 --sigma 0.25 --noise_choice GN --resume_path experiments_stegastamp/coco/OO/-2024-05-27-11:49-train/path_checkpoint/epoch_499_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 200 --num 10000 &
# CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num_processes 2 --main_process_port 29515 Stega_train.py --file options/Stega/mi/Dmi_opt.yml --dataset coco --batch_size 500 --sigma 0.5 --noise_choice GN --resume_path experiments_stegastamp/coco/OO/-2024-05-27-11:49-train/path_checkpoint/epoch_499_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 200 --num 10000
# CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num_processes 2 --main_process_port 29516 Stega_train.py --file options/Stega/mi/Dmi_opt.yml --dataset coco --batch_size 500 --sigma 0.03 --noise_choice affine --resume_path experiments_stegastamp/coco/OO/-2024-05-27-11:49-train/path_checkpoint/epoch_499_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 200 --num 10000



# 9.1 emperical with mi loss

# Celeb
# CUDA_VISIBLE_DEVICES=4,5 accelerate launch --num_processes 2 --main_process_port 29504 Stega_train.py --file options/Stega/mi/Dmi_opt.yml --dataset celeb --batch_size 480 --sigma 0.03 --noise_choice stega_combined --resume_path /mnt/shared/Huggingface/sharedcode/Stegastamp_Train/weights/CelebA/stega/emperical/epoch_499_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100
# # COCO
# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 --main_process_port 29505 Stega_train.py --file options/Stega/mi/Dmi_opt.yml --dataset coco --batch_size 500 --sigma 0.1 --noise_choice stega_combined --resume_path /mnt/shared/Huggingface/sharedcode/Stegastamp_Train/weights/COCO/stega/emperical/epoch_499_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --num 10000


# 11.21--------- STEGA IMAGENET
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 --main_process_port 29513 Stega_train.py --file options/Stega/mi/Dmi_opt.yml --dataset imagenet --batch_size 128 --sigma 0.1 --noise_choice GN --resume_path experiments_stegastamp/imagenet/OO/-2024-11-20-11:31-train/path_checkpoint/epoch_499_state.pth \
--lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --num 20000 
# CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 --main_process_port 29514 Stega_train.py --file options/Stega/mi/Dmi_opt.yml --dataset imagenet --batch_size 96 --sigma 0.25 --noise_choice GN --resume_path experiments_stegastamp/imagenet/OO/-2024-11-20-11:31-train/path_checkpoint/epoch_499_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --num 20000 
# CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 --main_process_port 29515 Stega_train.py --file options/Stega/mi/Dmi_opt.yml --dataset imagenet --batch_size 96 --sigma 0.5 --noise_choice GN --resume_path experiments_stegastamp/imagenet/OO/-2024-11-20-11:31-train/path_checkpoint/epoch_499_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --num 20000
# CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 --main_process_port 29516 Stega_train.py --file options/Stega/mi/Dmi_opt.yml --dataset imagenet --batch_size 96 --sigma 0.01 --noise_choice affine --resume_path experiments_stegastamp/imagenet/OO/-2024-11-20-11:31-train/path_checkpoint/epoch_499_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --num 20000
# CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 --main_process_port 29516 Stega_train.py --file options/Stega/mi/Dmi_opt.yml --dataset imagenet --batch_size 96 --sigma 0.02 --noise_choice affine --resume_path experiments_stegastamp/imagenet/OO/-2024-11-20-11:31-train/path_checkpoint/epoch_499_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --num 20000
# CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 --main_process_port 29516 Stega_train.py --file options/Stega/mi/Dmi_opt.yml --dataset imagenet --batch_size 96 --sigma 0.03 --noise_choice affine --resume_path experiments_stegastamp/imagenet/OO/-2024-11-20-11:31-train/path_checkpoint/epoch_499_state.pth \
# --lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --num 20000