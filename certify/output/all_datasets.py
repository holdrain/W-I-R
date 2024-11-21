import glob
import os
from time import time

import PIL
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


def celeb(batch_sz,path="/mnt/shared/deepfake/CelebA-HQ/val"):
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

def coco(batch_sz,path="/mnt/shared/coco2017/test2017"):
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

def ae(batch_sz,path="/data/shared/Huggingface/sharedcode/Stegastamp_CR/ae_data/G_A/0.1_0.02"):
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


class CustomImageFolder(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.filenames = glob.glob(os.path.join(data_dir, "*.png"))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpeg")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
        self.filenames = sorted(self.filenames)
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = PIL.Image.open(filename)
        if self.transform:
            image = self.transform(image)
        return image, 0

    def __len__(self):
        return len(self.filenames)
    
