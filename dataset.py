import glob
import os
from time import time

import PIL
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from utils.helpers import change_path

from setting import *


class CustomImageFolder(Dataset):
    def __init__(self, data_dir, transform=None,num=None):
        self.data_dir = data_dir
        self.filenames = glob.glob(os.path.join(data_dir, "*.png"))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpeg")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
        self.filenames = sorted(self.filenames)
        self.transform = transform
        self.num = num
        if self.num is not None:
            self.filenames = self.filenames[:self.num]

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = PIL.Image.open(filename).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0

    def __len__(self):
        return len(self.filenames)

def get_celebdl(args):
    transform_pipe = [
        transforms.Resize((args.data.image_resolution,args.data.image_resolution)),
        transforms.ToTensor(),
    ]
    if args.data.normalize:
        transform_pipe.append(transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]))
    transform = transforms.Compose(transform_pipe)

    s = time()
    data_dir = change_path(CELEBAHQ_TRAIN_PATH)
    print(f"Loading image folder {data_dir} ...")
    ds = CustomImageFolder(data_dir, transform=transform,num=args.data.num)
    print(f"Finished. Loading took {time() - s:.2f}s")
    train_size = int(args.train.train_ratio*len(ds))
    val_size = int(args.train.val_ratio*len(ds))
    tds,vds = random_split(ds,[train_size,val_size])
    print(f"Train_size:{train_size},Val_size:{val_size}")
    tdl = DataLoader(tds,batch_size=args['train']['batch_size'],shuffle=True,num_workers=args['train']['num_worker'],pin_memory=True)
    vdl = DataLoader(vds,batch_size=args['val']['batch_size'],shuffle=False,num_workers=4,pin_memory=True)
    return tdl,vdl

def get_cocodl(args):
    transform_pipe = [
        transforms.CenterCrop(148),
        transforms.Resize((args.data.image_resolution,args.data.image_resolution)),
        transforms.ToTensor(),
    ]
    if args.data.normalize:
        transform_pipe.append(transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]))
    transform = transforms.Compose(transform_pipe)

    s = time()
    train_dir = change_path(MSCOCO_TRAIN_PATH)
    val_dir = change_path(MSCOCO_VAL_PATH)
    print(f"Loading image folder {train_dir} and {val_dir}...")
    tds = CustomImageFolder(train_dir, transform=transform,num=args.data.num)
    vds = CustomImageFolder(val_dir, transform=transform,num=None)
    print(f"Finished. Loading took {time() - s:.2f}s")
    print(f"Train_size:{int(len(tds))},Val_size:{int(len(vds))}")
    tdl = DataLoader(tds,batch_size=args['train']['batch_size'],shuffle=True,num_workers=args['train']['num_worker'])
    vdl = DataLoader(vds,batch_size=args['val']['batch_size'],shuffle=False,num_workers=0)
    return tdl,vdl


def get_imagenet(args):
    transform_pipe = [
        transforms.Resize(300),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ]
    if args.data.normalize:
        transform_pipe.append(transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]))
    transform = transforms.Compose(transform_pipe)

    s = time()
    data_dir = change_path(IMAGENET_TRAIN_PATH)
    print(f"Loading image folder {data_dir} ...")
    ds = datasets.ImageFolder(data_dir, transform=transform)
    if args.data.num is not None:
        ds = Subset(ds, range(args.data.num))
    print(f"Finished. Loading took {time() - s:.2f}s")
    train_size = int(args.train.train_ratio*len(ds))
    val_size = int(args.train.val_ratio*len(ds))
    tds,vds = random_split(ds,[train_size,val_size])
    print(f"Train_size:{train_size},Val_size:{val_size}")
    tdl = DataLoader(tds,batch_size=args['train']['batch_size'],shuffle=True,num_workers=args['train']['num_worker'],pin_memory=True)
    vdl = DataLoader(vds,batch_size=args['val']['batch_size'],shuffle=False,num_workers=4,pin_memory=True)
    return tdl,vdl