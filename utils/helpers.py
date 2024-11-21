import os
import random
import sys
from os.path import join

import numpy as np
import torch
from easydict import EasyDict

import models.stega as stega
from models.hidden import Decoder, Encoder
from models.stega import StegaStampDecoder, StegaStampEncoder


def ber_on_str(str1, str2):
    assert len(str1) == len(str2), "string1 and string2 must share same length"
    different_bits_count = sum(1 for bit1, bit2 in zip(str1, str2) if bit1 != bit2)
    return float(different_bits_count/len(str1))

def read_txt_file(fpath):
    with open(fpath,'r') as file:
        lines = file.readlines()
    return [line.strip().split(' ')[1] for line in lines]

def msg2str(message):
    string = ''.join(str(int(i)) for i in message.view(-1))
    return string

def str2msg(str):
    return [True if el=='1' else False for el in str]

def cal_ber(emb_path,det_path):
    emb_fingerprints = read_txt_file(emb_path)
    det_fingerprints = read_txt_file(det_path)
    sum_ber = 0.0
    for f1,f2 in zip(emb_fingerprints,det_fingerprints):
        sum_ber += ber_on_str(f1,f2)
    return sum_ber/len(emb_fingerprints)

def new_dir(path):
    '''
    create a new folder if it not exists
    '''
    if not os.path.exists(path):
        os.makedirs(path,exist_ok=True)
    return path

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print 

def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def get_stega(args):
    '''
    return encoder and unwrapped decoder
    '''

    encoder = stega.StegaStampEncoder(args.image_resolution,args.Image_channels,args.message_length,
                                      return_residual=False,
                                      use_noise=args.encoder_noise,
                                      e_sigma=args.e_sigma,
                                      frozen_upsample=args.frozen_upsample,
                                      )
    decoder = stega.StegaStampDecoder(args.image_resolution,args.Image_channels,args.message_length,)
    if args.pretrained_e is not None and args.pretrained_d is not None:
        encoder_w = torch.load(args.pretrained_e,map_location='cpu')
        decoder_w = torch.load(args.pretrained_d,map_location='cpu')
        # unwrapped(no deformwrapper)
        decoder_w = {key.replace('base_decoder.', ''): value for key, value in decoder_w.items()}
        decoder_w = {key.replace('module.', ''): value for key, value in decoder_w.items()}
        unexpected_keys = encoder.load_state_dict(encoder_w,strict=False)
        print("uncompatible keys in encoder:",unexpected_keys)
        decoder.load_state_dict(decoder_w,strict=True)
        print("loading pretrained model success!")
    else:
        print("init model parameters randomly!")
    return encoder,decoder

def get_hiddenmodel(args):
    '''
    return encoder and unwrapped decoder(Hidden)
    '''
    encoder = Encoder(args)
    decoder = Decoder(args)
    if args.pretrained_e is not None and args.pretrained_d is not None:
        encoder_w = torch.load(args.pretrained_e,map_location='cpu')
        decoder_w = torch.load(args.pretrained_d,map_location='cpu')
        # unwrapped(no deformwrapper)
        decoder_w = {key.replace('base_decoder.', ''): value for key, value in decoder_w.items()}
        decoder_w = {key.replace('module.', ''): value for key, value in decoder_w.items()}
        unexpected_keys = encoder.load_state_dict(encoder_w,strict=False)
        print("uncompatible keys in encoder:",unexpected_keys)
        decoder.base_decoder.load_state_dict(decoder_w,strict=True)
        print("loading pretrained model success!")
    else:
        print("init model parameters randomly!")
    return encoder,decoder

def save_networks(i_epoch,path,state_dict):

    torch.save(state_dict,join(path, f"epoch_{i_epoch}_state.pth"))
    print("saving state of training!")
    return

def resume_stega(encoder,decoder,discriminator,ckp_path):
    state_dict = torch.load(ckp_path,map_location='cpu')
    encoder_w = state_dict["encoder"]
    decoder_w = state_dict["decoder"]
    steps_since_im_loss_activated = -1
    if discriminator is not None:
        steps_since_im_loss_activated = state_dict["steps_since_im_loss_activated"]
        discriminator_w = state_dict["Discriminator"]
        discriminator.load_state_dict(discriminator_w,strict=True)
    uncompatible_keys = encoder.load_state_dict(encoder_w,strict=False)
    print("resume encoder uncompatible keys:",uncompatible_keys)
    decoder.load_state_dict(decoder_w,strict=True)
    return encoder,decoder,discriminator,steps_since_im_loss_activated

def resume_hidden(encoder,decoder,discriminator,ckp_path):
    '''
    resume for hidden
    '''
    state_dict = torch.load(ckp_path,map_location='cpu')
    encoder_w = state_dict["encoder"]
    decoder_w = state_dict["decoder"]
    discriminator_w = state_dict["Discriminator"]
    uncompatible_keys = encoder.load_state_dict(encoder_w,strict=False)
    print("resume encoder uncompatible keys:",uncompatible_keys)
    decoder.load_state_dict(decoder_w,strict=True)
    discriminator.load_state_dict(discriminator_w,strict=True)
    return encoder,decoder,discriminator

def cal_tolerant(total_bits, beta=0.05):
    from scipy.stats import binom
    for k in range(total_bits + 1):
        prob = 1 - binom.cdf(k - 1, total_bits, 0.5)  # 计算概率
        if prob <= beta:
            break
    return total_bits - k

def change_path(path:str):
    if not os.path.exists(path):
        if path.startswith("/mnt"):
            path_ = "/data"+path[4:]
        else:
            path_ = "/mnt"+path[5:]
    else:
        path_ = path
    return path_

def load_weights(ckp_path,model,message_len):
    '''
    return pretrained encoder and decoder(unwrapped version)
    '''
    if model == "stega":
        image_resolution,Image_channels,message_length = (128,3,message_len)
        encoder = StegaStampEncoder(image_resolution,Image_channels,message_length)
        decoder = StegaStampDecoder(image_resolution,Image_channels,message_length)
        print("dd")
    elif model == "hidden":
        config = {
            'H': 128,
            'W': 128,
            'encoder_channels': 64,
            'encoder_blocks': 4,

            'decoder_channels': 64,
            'decoder_blocks': 7,

            'discriminator_channels': 64,
            'discriminator_blocks': 4,
            'message_length': message_len
        }
        config = EasyDict(config)
        encoder = Encoder(config)
        decoder = Decoder(config)
    else:
        raise NotImplementedError
    if type(ckp_path) == list:
        encoder_w = torch.load(ckp_path[0],map_location='cpu')
        decoder_w = torch.load(ckp_path[1],map_location='cpu')
    else:
        state_dict = torch.load(ckp_path,map_location='cpu')
        encoder_w = state_dict["encoder"]
        decoder_w = state_dict["decoder"]
    decoder_w = {key.replace('base_decoder.', ''): value for key, value in decoder_w.items()}
    decoder_w = {key.replace('module.', ''): value for key, value in decoder_w.items()}
    uncompatible_keys = encoder.load_state_dict(encoder_w,strict=False)
    print("uncompatible keys:",uncompatible_keys)
    decoder.load_state_dict(decoder_w,strict=True)

    return encoder,decoder
if __name__ == "__main__":
    import os
    attack_list = os.listdir("/data/shared/Huggingface/sharedcode/Stegastamp_CR/attack_results")
    for attack_l in attack_list:
        print(f"{attack_l} ber",cal_ber("/data/shared/Huggingface/sharedcode/Stegastamp_CR/embedding_outputs/embedded_fingerprints.txt",f"/data/shared/Huggingface/sharedcode/Stegastamp_CR/attack_results/{attack_l}/detected_fingerprints.txt"))