import math

import torch
import torch.nn.init as init
from torch import nn, sigmoid
from torch.nn.functional import relu


class StegaStampEncoder(nn.Module):
    def __init__(
        self,
        resolution=32,
        IMAGE_CHANNELS=1,
        fingerprint_size=100,
        return_residual=False,
        use_noise = False,
        bernoli_flips = False,
        frozen_upsample = False,
        bernoli_beta = 0.2,
        e_sigma = 0.2,
    ):
        super(StegaStampEncoder, self).__init__()
        self.fingerprint_size = fingerprint_size
        self.IMAGE_CHANNELS = IMAGE_CHANNELS
        self.return_residual = return_residual
        self.secret_dense = nn.Linear(self.fingerprint_size, 16 * 16 * IMAGE_CHANNELS)
        self.use_noise = use_noise
        self.e_sigma = e_sigma
        self.bernoli_flips = bernoli_flips
        self.bernoli_beta = bernoli_beta
        self.frozen_upsample = frozen_upsample

        log_resolution = int(math.log(resolution, 2))
        assert resolution == 2 ** log_resolution, f"Image resolution must be a power of 2, got {resolution}."

        self.fingerprint_upsample = nn.Upsample(scale_factor=(2**(log_resolution-4), 2**(log_resolution-4)))
        self.conv1 = nn.Conv2d(2 * IMAGE_CHANNELS, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 2, 1)
        self.pad6 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up6 = nn.Conv2d(256, 128, 2, 1)
        self.upsample6 = nn.Upsample(scale_factor=(2, 2))
        self.conv6 = nn.Conv2d(128 + 128, 128, 3, 1, 1)
        self.pad7 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up7 = nn.Conv2d(128, 64, 2, 1)
        self.upsample7 = nn.Upsample(scale_factor=(2, 2))
        self.conv7 = nn.Conv2d(64 + 64, 64, 3, 1, 1)
        self.pad8 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up8 = nn.Conv2d(64, 32, 2, 1)
        self.upsample8 = nn.Upsample(scale_factor=(2, 2))
        self.conv8 = nn.Conv2d(32 + 32, 32, 3, 1, 1)
        self.pad9 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up9 = nn.Conv2d(32, 32, 2, 1)
        self.upsample9 = nn.Upsample(scale_factor=(2, 2))
        self.conv9 = nn.Conv2d(32 + 32 + 2 * IMAGE_CHANNELS, 32, 3, 1, 1)
        self.conv10 = nn.Conv2d(32, 32, 3, 1, 1)
        self.residual = nn.Conv2d(32, IMAGE_CHANNELS, 1)
        if self.frozen_upsample:
            for _ , param in self.secret_dense.named_parameters():
                param.requires_grad = False

    def forward(self, fingerprint, image):
        if self.bernoli_flips:
            s = torch.bernoulli(self.bernoli_beta * torch.ones(self.fingerprint_size))
            fingerprint = torch.bitwise_xor(s,fingerprint)

        fingerprint = relu(self.secret_dense(fingerprint))
        fingerprint = fingerprint.view((-1, self.IMAGE_CHANNELS, 16, 16))
        fingerprint_enlarged = self.fingerprint_upsample(fingerprint)
        
        if self.use_noise:
            fingerprint_enlarged += torch.randn_like(fingerprint_enlarged)*self.e_sigma
        
        inputs = torch.cat([fingerprint_enlarged, image], dim=1)
        conv1 = relu(self.conv1(inputs))
        conv2 = relu(self.conv2(conv1))
        conv3 = relu(self.conv3(conv2))
        conv4 = relu(self.conv4(conv3))
        conv5 = relu(self.conv5(conv4))
        up6 = relu(self.up6(self.pad6(self.upsample6(conv5))))
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = relu(self.conv6(merge6))
        up7 = relu(self.up7(self.pad7(self.upsample7(conv6))))
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = relu(self.conv7(merge7))
        up8 = relu(self.up8(self.pad8(self.upsample8(conv7))))
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = relu(self.conv8(merge8))
        up9 = relu(self.up9(self.pad9(self.upsample9(conv8))))
        merge9 = torch.cat([conv1, up9, inputs], dim=1)
        conv9 = relu(self.conv9(merge9))
        conv10 = relu(self.conv10(conv9))
        output = self.residual(conv10)
        if not self.return_residual:
            output = sigmoid(output)
        fingerprint_reduce = fingerprint_enlarged
        residual_reduce = output - image
        return output,fingerprint_reduce,residual_reduce


class StegaStampDecoder(nn.Module):
    def __init__(self, resolution=32, IMAGE_CHANNELS=1, fingerprint_size=1):
        super(StegaStampDecoder, self).__init__()
        self.resolution = resolution
        self.IMAGE_CHANNELS = IMAGE_CHANNELS
        self.decoder = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, 32, (3, 3), 2, 1),  # 16
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # 8
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1),  # 4
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # 2
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), 2, 1),
            nn.ReLU(),
        )
        # kaiming initialization for relu network
        for name, param in self.decoder.named_parameters():
            if 'weight' in name:
                init.kaiming_uniform_(param.data, mode='fan_in', nonlinearity='relu')
            elif 'bias' in name:
                init.constant_(param.data, 0)
        self.dense = nn.Sequential(
            nn.Linear(resolution * resolution * 128 // 32 // 32, 512),
            nn.ReLU(),
            nn.Linear(512, fingerprint_size),
        )

    def forward(self, image):
        x = self.decoder(image)
        x = x.view(-1, self.resolution * self.resolution * 128 // 32 // 32)
        return self.dense(x)

class Stega_Discriminator(nn.Module):
    def __init__(self):
        super(Stega_Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 1, 3,)
            )

    def forward(self, x):
        x = self.model(x)
        output = torch.mean(x)
        return output, x
    
if __name__ == "__main__":
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    def generate_random_fingerprints(fingerprint_length, batch_size=1):
        z = torch.zeros((batch_size, fingerprint_length), dtype=torch.float).random_(0, 2)
        return z
    image = Image.open('embedding_outputs/0000100111101000101011001110110011000011010111110010000000100100101010111001101011011000001110111100/10987.png').convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    print(image.size())

    encoder = StegaStampEncoder(resolution=128,IMAGE_CHANNELS=3,fingerprint_size=100)
    encoder(generate_random_fingerprints(100),image)