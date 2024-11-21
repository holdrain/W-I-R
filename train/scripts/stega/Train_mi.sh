CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --num_processes 4 --main_process_port 29513 Stega_train.py --file options/Stega/mi/Dmi_opt.yml --dataset imagenet --batch_size 128 --sigma 0.1 --noise_choice GN --resume_path experiments_stegastamp/imagenet/OO/-2024-11-20-11:31-train/path_checkpoint/epoch_499_state.pth \
--lre 0.0001 --lrd 0.0001 --set_start_epoch 0 --epoch 100 --num 20000 
