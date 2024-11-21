import csv
import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

from dataset import CustomImageFolder
from models.deform import DeformWrapper
from models.stega import *
from utils.helpers import *
from utils.metrics import psnr_ssim


def generate_random_fingerprints(message_length, batch_size=4):
    z = torch.zeros((batch_size, message_length), dtype=torch.float).random_(0, 2)
    return z

def validate_stega(vdl,encoder,decoder,accelerator,args,tolerant):
    device = accelerator.device
    encoder.eval(),decoder.eval()
    all_bit_acc = []
    all_acc = []
    all_psnr = []
    all_ssim = []
    with tqdm(initial=0,total=len(vdl),position=0,desc="Validating....",disable=not accelerator.is_main_process) as pbar:
        for images,_ in vdl:
            with torch.inference_mode():
                messages = generate_random_fingerprints(args.model.message_length, images.size(0),)
                clean_images = images.to(device)
                messages = messages.to(device)

                watermarked_images,_,_ = encoder(messages, clean_images)
                decoder_output,noised_images = decoder(watermarked_images)
                messages_predicted = (decoder_output > 0).long()
                psnr,_,ssim = psnr_ssim(clean_images,watermarked_images,noised_images)
                difference = (messages_predicted != messages).float()
                correct = (difference.sum(dim=1)<=tolerant).float()
                bitwise_accuracy = (1.0 - difference.mean(dim=1))
                pbar.update(1)
            if accelerator.num_processes > 1:
                accelerator.wait_for_everyone()
                gathered_bit_acc = accelerator.gather_for_metrics(bitwise_accuracy) # (num_processes,batchsize)
                gathered_acc = accelerator.gather_for_metrics(correct) # (num_processes,batchsize)
                gathered_psnr = accelerator.gather_for_metrics(psnr) # (num_processes,batchsize)
                gathered_ssim = accelerator.gather_for_metrics(ssim) 
                all_bit_acc.append(gathered_bit_acc.mean())
                all_acc.append(gathered_acc.mean())
                all_psnr.append(gathered_psnr.mean().item())
                all_ssim.append(gathered_ssim.mean().item())
            else:
                all_bit_acc.append(bitwise_accuracy.mean().item())
                all_acc.append(correct.mean().item())
                all_psnr.append(psnr.mean().item())
                all_ssim.append(ssim.mean().item())
    all_bit_acc = torch.tensor(all_bit_acc)
    all_acc = torch.tensor(all_acc)
    all_psnr = torch.tensor(all_psnr)
    all_ssim= torch.tensor(all_ssim)
    return torch.mean(all_bit_acc).item(),torch.mean(all_acc).item(),torch.mean(all_psnr).item(),torch.mean(all_ssim).item()

def validate_hidden(vdl,encoder,decoder,accelerator,args,tolerant):
    device = accelerator.device
    encoder.eval(),decoder.eval()
    all_bit_acc = []
    all_acc = []
    all_psnr = []
    all_ssim = []
    with tqdm(initial=0,total=len(vdl),position=0,desc="Validating....",disable=not accelerator.is_main_process) as pbar:
        for images,_ in vdl:
            with torch.inference_mode():
                messages = generate_random_fingerprints(args.model.message_length, images.size(0),)
                clean_images = images.to(device)
                messages = messages.to(device)

                watermarked_images,_ = encoder(clean_images,messages)
                decoder_output,noised_images = decoder(watermarked_images,clean_images.clone())
                messages_predicted = decoder_output.round().clip(0, 1)
                psnr,_,ssim = psnr_ssim(clean_images,watermarked_images,noised_images)
                difference = (messages_predicted != messages).float()
                correct = (difference.sum(dim=1)<=tolerant).float()
                bitwise_accuracy = (1.0 - difference.mean(dim=1))
                pbar.update(1)
            if accelerator.num_processes > 1:
                accelerator.wait_for_everyone()
                gathered_bit_acc = accelerator.gather_for_metrics(bitwise_accuracy) # (num_processes,batchsize)
                gathered_acc = accelerator.gather_for_metrics(correct) # (num_processes,batchsize)
                gathered_psnr = accelerator.gather_for_metrics(psnr) # (num_processes,batchsize)
                gathered_ssim = accelerator.gather_for_metrics(ssim) 
                all_bit_acc.append(gathered_bit_acc.mean().item())
                all_acc.append(gathered_acc.mean().item())
                all_psnr.append(gathered_psnr.mean().item())
                all_ssim.append(gathered_ssim.mean().item())
            else:
                all_bit_acc.append(bitwise_accuracy.mean().item())
                all_acc.append(correct.mean().item())
                all_psnr.append(psnr.mean().item())
                all_ssim.append(ssim.mean().item())
    all_bit_acc = torch.tensor(all_bit_acc)
    all_acc = torch.tensor(all_acc)
    all_psnr = torch.tensor(all_psnr)
    all_ssim= torch.tensor(all_ssim)
    # print(all_acc.size(),all_bit_acc.size())
    return torch.mean(all_bit_acc).item(),torch.mean(all_acc).item(),torch.mean(all_psnr).item(),torch.mean(all_ssim).item()

def psnr_ssim_acc(vdl,encoder,decoder,device,tolerant,model_choice):
    '''return psnr,ssim,bit-acc,acc on a dataset(only run on a single gpu)'''
    encoder.eval(),decoder.eval()
    all_bit_acc = []
    all_acc = []
    all_psnr = []
    all_ssim = []
    with tqdm(initial=0,total=len(vdl),position=0,desc="Validating....") as pbar:
        for images,_ in vdl:
            with torch.inference_mode():
                messages = generate_random_fingerprints(100, images.size(0),)
                clean_images = images.to(device)
                messages = messages.to(device)
                if model_choice == 'stega':
                    watermarked_images,_,_ = encoder(messages, clean_images)
                    decoder_output,noised_images = decoder(watermarked_images)
                    messages_predicted = (decoder_output > 0).long()
                else:
                    watermarked_images,_ = encoder(clean_images,messages)
                    decoder_output,noised_images = decoder(watermarked_images)
                    messages_predicted = decoder_output.round().clip(0,1).long()
                difference = (messages_predicted != messages).float()
                correct = (difference.sum(dim=1)<=tolerant).float()
                bitwise_accuracy = (1.0 - difference.mean(dim=1))
                psnr_wm2co,_,ssim_wm2co = psnr_ssim(clean_images,watermarked_images,noised_images)
                pbar.update(1)
                all_bit_acc.append(bitwise_accuracy.mean().item())
                all_acc.append(correct.mean().item())
                all_psnr.append(psnr_wm2co)
                all_ssim.append(ssim_wm2co)

    bit_acc = torch.mean(torch.tensor(all_bit_acc)).item()
    acc = torch.mean(torch.tensor(all_acc)).item()
    psnr = torch.mean(torch.tensor(all_psnr)).item()
    ssim = torch.mean(torch.tensor(all_ssim)).item()
    print(f"bit-acc:{bit_acc}--------acc:{acc}--------psnr:{psnr}----ssim:{ssim}")
    return bit_acc,acc,psnr,ssim


if __name__ == "__main__":
    set_seeds(2024)
    device = torch.device("cuda:0")
    for dataset in ["CelebA"]:
        for model_choice in ["hidden"]:
            message_len = 100
            mi = ""
            data_dir = {"CelebA":"/data/shared/deepfake/CelebA-HQ/val","COCO":"/data/shared/coco2017/test2017"}
            data_num = {"CelebA":None,"COCO":1000}
            ckp_root = f"/data/shared/Huggingface/sharedcode/Stegastamp_Train/{mi}weights/{dataset}/{model_choice}"
            
            transform_pipe = [
                transforms.Resize((128,128)),
                transforms.ToTensor(),
            ]
            if model_choice == 'hidden':
                transform_pipe.append(transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]))
            transform = transforms.Compose(transform_pipe)

            results_path = "./newresults" +'/' + dataset +'/'+ model_choice + f"/{mi}test.csv"
            batch_size = 300
            if dataset == "CelebA":
                ds = ImageFolder(data_dir[dataset],transform=transform)
            else:
                ds = CustomImageFolder(data_dir[dataset],transform,num=data_num[dataset])

            dl = DataLoader(ds,batch_size=batch_size,shuffle=False)
            # ckp_list = sorted(os.listdir(ckp_root))
            ckp_list = [f"{mi}emperical"]
            for op in ckp_list:
                if op == "OO":  
                    ckp = os.path.join(ckp_root,op,"epoch_499_state100.pth")
                elif op == "emperical":
                    ckp = os.path.join(ckp_root,op,"epoch_499_state.pth")
                else:
                    ckp = os.path.join(ckp_root,op,"epoch_99_state.pth")
                sigma = 0
                encoder,decoder = load_weights(ckp,model_choice,message_len)
                decoder = DeformWrapper(decoder,device,aug_method='OO',sigma=sigma)
                encoder,decoder = encoder.to(device),decoder.to(device)
                bit_acc,acc,psnr,ssim = psnr_ssim_acc(dl,encoder,decoder,device,tolerant=cal_tolerant(100),model_choice=model_choice)
                csv_dict = {"model":op,**{"Bit_acc(test)":bit_acc,"acc(test)":acc,"PSNR(test)":psnr,"SSIM(test)":ssim}}
                with open(results_path, mode='a', newline='') as file:
                    fieldnames = csv_dict.keys()
                    writer = csv.DictWriter(file, fieldnames=fieldnames)
                    if file.tell() == 0:
                        writer.writeheader()
                    writer.writerow(csv_dict)
                decoder = decoder.base_decoder