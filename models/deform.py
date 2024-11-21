import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.append("/mnt/shared/Huggingface/sharedcode/Stegastamp_Train")
import math
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

from models.hidden_noiser import Noiser
from models.noise_layers.crop import Crop
from models.noise_layers.cropout import Cropout
from models.noise_layers.dropout import Dropout
from models.noise_layers.resize import Resize
from utils.stegautils import *


class GaussianNoise(nn.Module):
    def __init__(self, std=0.1):
        super(GaussianNoise, self).__init__()
        self.std = std

    def forward(self, x):

        noise = torch.randn_like(x) * self.std
        noisy_img = x + noise
        noisy_img = torch.clamp(noisy_img, min=0, max=1)
        return noisy_img
    
class IdentityModel(nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()
        self.identity = nn.Identity()

    def forward(self, x):
        return self.identity(x)

class DeformWrapper(nn.Module):
    def __init__(self, model, device, aug_method, sigma=0.1, num_bases=2):
        super(DeformWrapper, self).__init__()
        self.base_decoder = model
        self.device = device
        self.aug_method = aug_method
        # We assume that the input is always between 0 and 1. For rotation, we do this normalization internally
        self.sigma = sigma*math.pi if aug_method =='rotation' else sigma
        self.num_bases = num_bases
        self.deformed_images = None
        if self.aug_method == 'hidden_combined':
            self.crop = Crop((0.4,0.55),(0.4,0.55))
            self.cropout = Cropout((0.15,0.25),(0.15,0.25))
            self.dropout = Dropout([0.15,0.35])
            self.resize =  Resize([0.4,0.6])
            self.noiser = Noiser([self.resize,self.cropout,self.crop,self.dropout],self.device)
        
    def _GenImageBlur(self,x):
        '''Stegastamp augmentation'''
        N_blur = 7
        f = random_blur_kernel(probs=[.25, .25], N_blur=N_blur, sigrange_gauss=[1., 3.], sigrange_line=[.25, 1.],
                                    wmin_line=3)
        f = f.to(self.device)
        return F.conv2d(x, f, bias=None, padding=int((N_blur - 1) / 2))
    
    def _contrastandbrightness(self,x):
        '''Stegastamp augmentation'''
        contrast_high = 1.5
        contrast_low = 0.5
        rnd_brightness = get_rnd_brightness_torch(0.3, 0.1, x.size(0))
        contrast_params = [contrast_low, contrast_high]
        contrast_scale = torch.Tensor(x.size()[0]).uniform_(contrast_params[0], contrast_params[1])
        contrast_scale = contrast_scale.reshape(x.size()[0], 1, 1, 1)
        contrast_scale = contrast_scale.to(self.device)
        rnd_brightness = rnd_brightness.to(self.device)
        x = x * contrast_scale
        x = x + rnd_brightness
        return torch.clamp(x, 0, 1)

    def _saturation(self,x):
        rnd_sat = torch.rand(1)[0] * 1.0
        sat_weight = torch.FloatTensor([.3, .6, .1]).reshape(1, 3, 1, 1)
        sat_weight = sat_weight.to(self.device)
        encoded_image_lum = torch.mean(x * sat_weight, dim=1).unsqueeze_(1)
        return (1 - rnd_sat) * x + rnd_sat * encoded_image_lum
    
    def _Jpeg(self,x):
        jpeg_quality = torch.rand(1)[0] * (100. - 50)
        x = x.reshape([-1, 3, 128, 128])
        return jpeg_compress_decompress(x, rounding=round_only_at_0,quality=jpeg_quality).contiguous()

    def _Noise(self,x):
        rnd_noise = torch.rand(1)[0] * 0.2
        noise = torch.normal(mean=0, std=rnd_noise, size=x.size(), dtype=torch.float32)
        noise = noise.to(self.device)
        x = x + noise
        return torch.clamp(x, 0, 1)

    def _GenImageAffine(self, x):
        N, _, rows, cols = x.shape # N is the batch size
        params = torch.randn((6, N, 1, 1)) * self.sigma

        #Generating the vector field for Affine transformation.
        X, Y = torch.meshgrid(torch.linspace(-1,1,rows),torch.linspace(-1,1,cols))
        X, Y = X.unsqueeze(0), Y.unsqueeze(0)
        Xv = params[0]*X + params[1]*Y + params[2]
        Yv = params[3]*X + params[4]*Y + params[5]
        
        randomFlow = torch.stack((Yv,Xv), axis=3).to(self.device)
        grid = torch.stack((Y,X), axis=3).to(self.device)
        
        return F.grid_sample(x, grid+randomFlow,align_corners=True)

    def forward(self,x,cover=None):
        if self.aug_method == 'stega_combined':
            # aug_method = self._GenImageBlur
            # aug_method = self._contrastandbrightness
            # aug_method = self._saturation
            # aug_method = self._Jpeg
            aug_method = random.choice([self._GenImageBlur,self._Noise,self._contrastandbrightness,self._saturation])
            x = aug_method(x)
        elif self.aug_method == 'hidden_combined':
            x = self.noiser([x,cover])[0]
        elif self.aug_method == 'OO':
            pass
        elif self.aug_method == 'affine':
            x = self._GenImageAffine(x)
        elif self.aug_method == "GN":
            x = x + torch.randn_like(x) * self.sigma
        else:
            raise Exception("Un identified augmentation method!")
        self.deformed_images = x
        # print(x.size())
        return self.base_decoder(x),self.deformed_images


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from PIL import Image
    from torchvision import transforms

    from models.stega import StegaStampDecoder

    # 加载图片并转换为Tensor  
    def load_image_to_tensor(image_path):  
        image = Image.open(image_path).convert('RGB')  # 确保是RGB图像  
        transform = transforms.ToTensor()  
        tensor_image = transform(image)  
        return tensor_image  
    
    # 主函数  
    image_path = '/mnt/shared/deepfake/CelebA-HQ/resized128/val/female/049795.jpg'  # 替换为你的图片路径  
    tensor_image = load_image_to_tensor(image_path)
    decoder = StegaStampDecoder(128,3,100)
    wraped_decoder = DeformWrapper(decoder, torch.device('cpu'), aug_method='hidden_combined')
    _,aug_image = wraped_decoder(tensor_image.unsqueeze(0),tensor_image.unsqueeze(0))
    # 将Tensor转换为numpy数组并调整形状  
    np_image = aug_image.squeeze(0).permute(1, 2, 0).detach().numpy()  # [C, H, W] -> [H, W, C]  

    plt.imshow(np_image)  
    plt.axis('off')  # 不显示坐标轴  
    plt.show()  