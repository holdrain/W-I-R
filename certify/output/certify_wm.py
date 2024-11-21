import argparse
import datetime
import math
import os
from time import time

import all_datasets
import torch
from easydict import EasyDict
from tqdm.auto import tqdm
from WM_smooth import WMSmooth

from models.hidden import Decoder, Encoder
from models.stega import *
from utils.certifyutils import *

dataset_choices = ['coco', 'celeb','ae']
model_choices = ['stega', 'hidden']
certification_method_choices = ['OO','GN','rotation', 'affine', 'scaling_uniform']
message_lengths = {'stega':100,'hidden':100}


parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("--dataset", choices=dataset_choices, help="which dataset")
parser.add_argument("--model", type=str, choices=model_choices, help="model name")
parser.add_argument("--base_model", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("--certify_method", type=str, default='rotation', required=True, choices=certification_method_choices, help='type of certification for certification')
parser.add_argument("--sigma", type=float, help="noise hyperparameter")
parser.add_argument("--experiment_name", type=str, required=True,
                    help='name of directory for saving results')
parser.add_argument("--certify_batch_sz", type=int, default=400, help="cetify batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
parser.add_argument('--uniform', action='store_true', default=False, help='certify with uniform distribution')
parser.add_argument('--start',type=int,default=0)
parser.add_argument('--end',type=int,default=100)

args = parser.parse_args()

# seed
set_seeds(2024)
# full path for output
args.basedir = os.path.join('output/certify', args.dataset, args.model, args.experiment_name)
# Log path: verify existence of output_path dir, or create it
if not os.path.exists(args.basedir):
    os.makedirs(args.basedir, exist_ok=True)
args.outfile = os.path.join(args.basedir, f'result{args.start}-{args.end}.txt')
device = torch.device("cuda:0")
def load_weights(ckp_path,model):
    '''
    return pretrained encoder and decoder(unwrapped version)
    '''
    if model == "stega":
        image_resolution,Image_channels,message_length = (128,3,100)
        encoder = StegaStampEncoder(image_resolution,Image_channels,message_length)
        decoder = StegaStampDecoder(image_resolution,Image_channels,message_length)
        state_dict = torch.load(ckp_path,map_location='cpu')
        encoder_w = state_dict["encoder"]
        decoder_w = state_dict["decoder"]
        uncompatible_keys = encoder.load_state_dict(encoder_w,strict=False)
        print("uncompatible keys:",uncompatible_keys)
        decoder.load_state_dict(decoder_w,strict=True)
    elif model == 'hidden':
        config = {
            'H': 128,
            'W': 128,
            'encoder_channels': 64,
            'encoder_blocks': 4,

            'decoder_channels': 64,
            'decoder_blocks': 7,

            'discriminator_channels': 64,
            'discriminator_blocks': 4,
            'message_length': 100
        }
        config = EasyDict(config)
        encoder = Encoder(config)
        decoder = Decoder(config)
        state_dict = torch.load(ckp_path,map_location='cpu')
        encoder_w = state_dict["encoder"]
        decoder_w = state_dict["decoder"]
        uncompatible_keys = encoder.load_state_dict(encoder_w,strict=False)
        print("uncompatible keys:",uncompatible_keys)
        decoder.load_state_dict(decoder_w,strict=True)

    return encoder,decoder

def generate_random_message(message_length, batch_size=1):
    z = torch.zeros((batch_size, message_length), dtype=torch.float).random_(0, 2)
    return z

if __name__ == "__main__":
    # load dataset
    if hasattr(all_datasets, args.dataset):
        get_data_loaders = getattr(all_datasets, args.dataset)
        test_loader = get_data_loaders(1) # process an image at a time
    else:
        raise Exception('Undefined Dataset')

    # load model
    encoder,decoder = load_weights(args.base_model,args.model)
    encoder.eval(),decoder.eval()
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    if args.certify_method == 'rotation':
        args.sigma *= math.pi # For rotaions to transform the angles to [0, pi]

    # create the smooothed classifier g
    tolerant = cal_tolerant(message_lengths[args.model])
    smoothed_decoder = WMSmooth(args.model,decoder,tolerant,args.sigma,args.certify_method,device)
    
    # prepare output file
    f = open(args.outfile, 'w')
    print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = test_loader.dataset
    # print(len(dataset.filenames))
    with tqdm(total=args.end-args.start) as pbar:
        for i in range(args.start,args.end):
            # only certify every args.skip examples, and stop after args.max examples
            if i % args.skip != 0:
                continue
            if i == args.max:
                break

            (x, _) = dataset[i]

            before_time = time()
            # certify the prediction of g around x
            x = x.to(device)
            message = generate_random_messages(message_lengths[args.model]) # (1,message_length)
            message_str = msg2str(message[0])
            message = message.to(device)
            if args.model == 'stega':
                wm_x,_,residual = encoder(message,x.unsqueeze(0))
            else:
                wm_x,residual = encoder(x.unsqueeze(0),message)
            wm_x = wm_x.squeeze(0)
            # prediction, radius, p_A , error_bits = smoothed_decoder.certify(wm_x, args.N0, args.N, args.alpha, args.certify_batch_sz,message_str)
            # scheme zp
            prediction, radius, p_A = smoothed_decoder.certify(wm_x, args.N0, args.N, args.alpha, args.certify_batch_sz,message_str)
            if args.certify_method == "scaling_uniform":
                radius = 2 * smoothed_decoder.sigma * (p_A - 0.5)
            after_time = time()
            correct = prediction
            print('Time for certifying one image is', after_time - before_time )
            time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))

            # zp scheme
            print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
                i, message_str, prediction, radius, correct,time_elapsed), file=f, flush=True)

            pbar.update(1)

        f.close()
