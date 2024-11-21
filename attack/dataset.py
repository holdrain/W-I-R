import os
import glob
import PIL
from time import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

class CustomImageFolder(Dataset):
    def __init__(self, data_dir, transform=None,num=None):
        self.data_dir = data_dir
        self.filenames = glob.glob(os.path.join(data_dir, "*.png"))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpeg")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
        self.filenames = sorted(self.filenames)[:num]
        if num is not None:
            self.filenames = self.filenames[:num]
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = PIL.Image.open(filename)
        if self.transform:
            image = self.transform(image)
        return image, 0

    def __len__(self):
        return len(self.filenames)

def get_ds(args):
    if args.use_celeba_preprocessing:
        assert args.image_resolution == 128, f"CelebA preprocessing requires image resolution 128, got {args.image_resolution}."
        transform = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(128),
                transforms.ToTensor(),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(args.image_resolution),
                transforms.ToTensor(),
            ]
        )

    s = time()
    print(f"Loading image folder {args.data_dir} ...")
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        if args.data_dir.startswith('/mnt'):
            data_dir = args.data_dir.replace('/mnt', '/data', 1)
        elif args.data_dir.startswith('/data'):
            data_dir = args.data_dir.replace('/data', '/mnt', 1)
    dataset = CustomImageFolder(data_dir, transform=transform)
    print(f"Finished. Loading took {time() - s:.2f}s")
    return dataset