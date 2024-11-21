
import os
from time import time

from dataset import CustomImageFolder
from setting import *
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def celeb(batch_sz,path=CELEBAHQ_VAL_PATH):
    '''
    return dataloader of celeba-HQ
    '''
    transform = transforms.Compose(
        [
            transforms.Resize(128),
            transforms.ToTensor(),
        ]
    )
    s = time()
    print(f"Loading image folder {path} ...")
    data_dir = path
    if not os.path.exists(data_dir):
        if path.startswith('/mnt'):
            data_dir = path.replace('/mnt', '/data', 1)
        elif path.startswith('/data'):
            data_dir = path.replace('/data', '/mnt', 1)
    dataset = ImageFolder(data_dir, transform=transform)
    print(f"Finished. Loading took {time() - s:.2f}s")
    dl = DataLoader(dataset,batch_size=batch_sz,shuffle=False,num_workers=0)
    return dl

def coco(batch_sz,path=MSCOCO_TEST_PATH):
    '''
    return dataloader of mscoco
    '''
    transform = transforms.Compose(
        [
            transforms.CenterCrop(148),
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
        ]
    )
    s = time()
    print(f"Loading image folder {path} ...")
    data_dir = path
    if not os.path.exists(data_dir):
        if path.startswith('/mnt'):
            data_dir = path.replace('/mnt', '/data', 1)
        elif path.startswith('/data'):
            data_dir = path.replace('/data', '/mnt', 1)
    dataset = CustomImageFolder(data_dir, transform=transform)
    print(f"Finished. Loading took {time() - s:.2f}s")
    dl = DataLoader(dataset,batch_size=batch_sz,shuffle=False,num_workers=0)
    return dl

def ae(batch_sz,path="../attack/ae_data/G_A/0.1_0.02"):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    s = time()
    print(f"Loading image folder {path} ...")
    data_dir = path
    if not os.path.exists(data_dir):
        if path.startswith('/mnt'):
            data_dir = path.replace('/mnt', '/data', 1)
        elif path.startswith('/data'):
            data_dir = path.replace('/data', '/mnt', 1)
    dataset = CustomImageFolder(data_dir, transform=transform)
    print(f"Finished. Loading took {time() - s:.2f}s")
    dl = DataLoader(dataset,batch_size=batch_sz,shuffle=False,num_workers=0)
    return dl


    
