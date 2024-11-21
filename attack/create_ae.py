import argparse

import torch
from dataset import CustomImageFolder
from models.deform import DeformWrapper
from setting import *
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from tqdm.auto import tqdm
from utils.helpers import *


def generate_random_fingerprints(fingerprint_length, batch_size=1):
    z = torch.zeros((batch_size, fingerprint_length), dtype=torch.float).random_(0, 2)
    return z

def create_ae(vdl,encoder,decoder,args,device,num,f):
    '''
    create a set of adversarial examples by deforming images in vdl
    encoder: no noise version
    decoder: deformwrapped version
    num: the goal num
    '''
    encoder.eval(),decoder.eval()
    count,patient = 0,0
    while count < num:
        with tqdm(initial=0,total=len(vdl),position=0,desc="creating ae....") as pbar:
            for image,_ in vdl:
                message = generate_random_fingerprints(args.message_len)
                clean_images = image.to(device)
                message = message.to(device)
                watermarked_images,_,_ = encoder(message, clean_images)
                decoder_output,_ = decoder(watermarked_images,args.sigma)
                if args.model_choice == 'stega':
                    message_prediction = (decoder_output > 0).long()
                else:
                    message_prediction = (decoder_output.round().clip(0, 1)).long()
                bit_error = torch.sum((message_prediction != message)).long()
                if bit_error > args.tolerant and count < num:
                    # save ae
                    ae = decoder.deformed_images
                    ae_path = os.path.join(args.output_dir,args.aug_method,str(args.sigma),f"{count}.png")
                    save_image(ae,ae_path,normalize=True)
                    print(f"{ae_path}\t{msg2str(message)}\t{msg2str(message_prediction)}\t{int(bit_error)}",file=f,flush=True)
                    count += 1
                if count >= num:
                    break
                pbar.update(1)
        patient += 1
        if patient > 100:
            print("cannot find ae!")
            break


if __name__ == "__main__":
    set_seeds(2024)
    data_dir = {
        'celeb': CELEBAHQ_VAL_PATH,
        'coco': MSCOCO_TEST_PATH
    }
    parser = argparse.ArgumentParser(description="Find ae")

    # data
    parser.add_argument("--data_choice", type=str,choices=['celeb','coco'])
    parser.add_argument("--model_choice",type=str,choices=['hidden','stega'])
    parser.add_argument("--message_len",type=int,default=100,help="Number of bits in the fingerprint.")
    parser.add_argument("--num_workers",type=int,default=0,help="the num of threads in dataloading for each process")
    parser.add_argument("--output_dir",type=str,default="ae_data",help="saving ae")
    parser.add_argument("--device",type=int,default=7)
    parser.add_argument("--seed",type=int,default=2024)
    parser.add_argument("--ckp_path",type=str,default="",help="pretrained model path when continue_train is False otherwise model saved in previous training")
    parser.add_argument("--aug_method",type=str,required=True)
    parser.add_argument("--num",type=int,default="ae nums")
    parser.add_argument("--sigma",type=float,nargs="+",default=0.1,help="sigma")
    args = parser.parse_args()

    set_seeds(args.seed)
    encoder,decoder = load_weights(args.ckp_path,args.model_choice,args.message_len)
    args.tolerant = cal_tolerant(args.message_len)
    new_dir(os.path.join(args.output_dir,args.aug_method,str(args.sigma)))

    transform_pipe = [
        transforms.Resize((image_resolution[args.data_choice],image_resolution[args.data_choice])),   
        transforms.ToTensor(),
    ]
    if args.model_choice == 'hidden':
        transform_pipe.append(transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]))
    transform = transforms.Compose(transform_pipe)
    if args.data_choice == "celeb":
        ds = ImageFolder(data_dir[args.data_choice],transform=transform)
    else:
        ds = CustomImageFolder(data_dir[args.data_choice],transform)
    dl = DataLoader(ds,batch_size=1,shuffle=False)
    decoder = DeformWrapper(decoder,args.device,args.aug_method)
    encoder,decoder = encoder.to(args.device),decoder.to(args.device)
    f = open(os.path.join(args.output_dir,args.aug_method,str(args.sigma),"label.txt"), 'w')
    print("path\tlabel\tpredict\terror_bit", file=f, flush=True)
    create_ae(dl,encoder,decoder,args,args.device,args.num,f)