import argparse
import csv
import os
import time
import warnings

import torch
import torch.multiprocessing as mp
from accelerate import Accelerator
from box import Box
from torch.optim import Adam, RMSprop
from torchvision.utils import save_image
from tqdm import tqdm
from tqdm.auto import tqdm

from dataset import get_celebdl, get_cocodl
from models.deform import DeformWrapper
from models.hidden import *
from utils.helpers import *
from utils.Hidden_trainer import Trainer
from utils.log import *
from utils.metrics import BatchMetricsAccumulator
from utils.yml import *
from val import validate_hidden

warnings.filterwarnings('ignore')

def generate_random_message(message_length, batch_size=4):
    z = torch.zeros((batch_size, message_length), dtype=torch.float).random_(0, 2)
    return z

def main(config):
    # reading config
    yml_path = config.file

    option_yml = parse_yml(yml_path)
    # convert to NoneDict, which returns None for missing keys
    opt = dict_to_nonedict(option_yml)
    opt = Box(opt)
    opt = update_param(opt,config)
    set_seeds(opt.train.seed)

    # cudnn
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # accelerate
    mp.set_start_method("spawn")
    accelerator = Accelerator(
        dispatch_batches=False,split_batches=False,
    )
    setup_for_distributed(accelerator.is_main_process)
    device = accelerator.device

    # log folder
    name = opt.name+opt.noise.choice
    if opt.noise.sigma is not None:
        name += ('_'+ str(opt.noise.sigma)) 
    time_now_NewExperiment = time.strftime("%Y-%m-%d-%H:%M", time.localtime())
    if opt.log.subfolder != None:
        subfolder_name = opt.log.subfolder + '/-'
    else:
        subfolder_name = ''
    folder_str = opt.log.logs_folder + f'/{opt.data.dataset}'+'/' + name + '/' + subfolder_name + \
        str(time_now_NewExperiment) + '-' + opt['train/test']
    opt_folder = folder_str  + '/opt'
    path_checkpoint = folder_str  + '/path_checkpoint'
    train_log_path = folder_str  + '/opt'+'/'+'train_log.csv'
    save_image_path = folder_str + '/images'

    # log option_yml
    new_dir(opt_folder)
    new_dir(path_checkpoint)
    new_dir(save_image_path)
    setup_logger('base', opt_folder, 'train_' + name, level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(dict2str(opt))

    # dl,model
    if opt.data.dataset == "celeb":
        tdl,vdl = get_celebdl(opt)
    else:
        tdl,vdl = get_cocodl(opt)
    # base model and optimizer
    encoder,decoder = get_hiddenmodel(opt.model)

    discriminator = Discriminator(opt.model)
    optimizer_en = Adam(params=encoder.parameters(), lr=opt.lr)
    optimizer_de = Adam(params=decoder.parameters(), lr=opt.lr)
    optimizer_dis = RMSprop(params=list(discriminator.parameters()), lr=0.00001)


    # train prepare
    total_epochs = opt['train']['epoch']
    start_epoch = opt['train']['set_start_epoch']
    
    if opt.resume:
        encoder,decoder,discriminator = resume_hidden(encoder,decoder,discriminator,opt.resume_path)
    
    # wrap decoder
    decoder = DeformWrapper(decoder,accelerator.device,aug_method=opt.noise.choice,sigma=opt.noise.sigma)
    tdl,vdl,encoder,decoder,discriminator,optimizer_en,optimizer_de,optimizer_dis = accelerator.prepare(tdl,vdl,
                                                                                                encoder,decoder,discriminator,
                                                                                                optimizer_en,optimizer_de,optimizer_dis)
    trainer = Trainer(opt,optimizer_en,optimizer_de,optimizer_dis,encoder,decoder,discriminator,device,accelerator)
    
    # base train loop
    metrics_updater = BatchMetricsAccumulator('hidden')
    for i_epoch in range(start_epoch,total_epochs):
        with tqdm(initial=0,total=len(tdl),position=0,desc=f"{name}-epoch:{i_epoch}",disable=not accelerator.is_main_process) as pbar:
            for images, _ in tdl:
                batch_size = min(opt.train.batch_size, images.size(0))
                messages = generate_random_message(opt.model.message_length, batch_size,)
                clean_images = images.to(device)
                messages = messages.to(device)

                b_loss_list,b_metric_list,image_dict = trainer.train_on_batch(clean_images,messages)
                metrics_updater.update(b_loss_list,b_metric_list)
                pbar.update(1)

        loss_dict,metric_dict = metrics_updater.compute_epoch_metric(method='mean')
        metrics_updater.reset()
        print("Bitwise accuracy(train){}".format(metric_dict['bit-acc']))
        print("PSNR(train): PSNR(W-C):{:.2f}---PSNR(N-W):{:.2f}---SSIM(W-C):{:.2f}".format(metric_dict['PSNR(W-C)'],
                                                                                        metric_dict['PSNR(N-W)'],
                                                                                          metric_dict['SSIM(W-C)']))
        # validate
        
        # tolerant = cal_tolerant(opt.model.message_length)
        # Bit_acc,acc,psnr,ssim = validate_hidden(vdl,encoder,decoder,accelerator,opt,tolerant)
        # print("Bit_acc(val){}".format(Bit_acc),"Acc(val):",acc)

        # Logging
        if accelerator.is_main_process:
            # train log
            # csv_dict = {"epoch":i_epoch,**loss_dict,**metric_dict,**{"Bit_acc(val)":Bit_acc,"PSNR(val)":psnr,"SSIM(val)":ssim}}
            csv_dict = {"epoch":i_epoch,**loss_dict,**metric_dict}
            with open(train_log_path, mode='a', newline='') as file:
                fieldnames = csv_dict.keys()
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                if file.tell() == 0:
                    writer.writeheader()
                writer.writerow(csv_dict)
            # save images
            # for key, tensor in image_dict.items():
            #     save_path = os.path.join(save_image_path, f"{key}_{i_epoch}.jpg")
            #     save_image(tensor, save_path)
                
            # checkpoint
            if (i_epoch+1) % opt.train.save_fre == 0:
                state_dict = {
                    "encoder":accelerator.unwrap_model(encoder).state_dict(),
                    "decoder":accelerator.unwrap_model(decoder).base_decoder.state_dict(),
                    "Discriminator":accelerator.unwrap_model(discriminator).state_dict(),
                }
                save_networks(i_epoch,path=path_checkpoint,state_dict=state_dict)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='option file')
    parser.add_argument('--file',type=str)
    parser.add_argument('--dataset',choices=['celeb','coco'])
    parser.add_argument('--num',type=int)
    parser.add_argument('--noise_choice',choices=['OO','GN','rotation','affine','hidden_combined'])
    parser.add_argument('--sigma',type=float)
    parser.add_argument('--batch_size',type=int)
    parser.add_argument('--resume_path',type=str)
    parser.add_argument('--lr',type=float)
    parser.add_argument('--epoch',type=int)
    parser.add_argument('--set_start_epoch',type=int)
    parser.add_argument('--optimizer',type=str)
    args = parser.parse_args()
    main(args)
                

