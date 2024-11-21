import torch
from torch import nn
from utils.vgg_loss import VGGLoss
from utils.metrics import psnr_ssim
import numpy as np
from torchvision.utils import make_grid
from models.mi_estimator import CLUBForCategorical

class Trainer():
    def __init__(self,opt,optimizer_en,optimizer_de,optimizer_dis,encoder,decoder,discriminator,device,accelerator) -> None:
        # setting
        self.opt = opt
        self.optimizer_en = optimizer_en
        self.optimizer_de = optimizer_de
        self.optimizer_dis = optimizer_dis
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.device = device
        self.accelerator = accelerator

        # fn
        self.l2_loss_fn = nn.MSELoss()
        self.BCE_loss_fn = nn.BCEWithLogitsLoss()
        self.vgg_loss_fn = VGGLoss(3,1,False).to(self.device)
        self.mse_loss = nn.MSELoss()

        if opt.train.mi_loss_type == 'kl' or opt.train.mi_loss_type == 'vsd':
            self.mi_fn = nn.KLDivLoss(reduction='batchmean')
            self.mi_loss_weight = opt.loss.mi_loss_weight
        elif opt.train.mi_loss_type == 'club':
            self.mi_fn = CLUBForCategorical(decoder=self.decoder)
            self.mi_loss_weight = opt.loss.mi_loss_weight
        else:
            self.mi_loss_weight = 0
        
    def train_on_batch(self,clean_images,messages):
        '''
        out_dict: {clean images, watermarked images, tup, message_prediction,D_output_real,D_output_fake}
        message: gt
        '''

        torch.autograd.set_detect_anomaly(True)
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()


        batch_size = clean_images.size(0)
        encoded_images,_ = self.encoder(clean_images, messages)
        residual = encoded_images - clean_images
        decoded_messages,noised_images = self.decoder(encoded_images,clean_images.clone())

        D_loss = torch.tensor(0.0)
        g_loss = torch.tensor(0.0)
        mi_loss = torch.tensor(0.0)

        # Discriminator
        d_target_label_cover = torch.full((batch_size, 1), 1, device=self.device).float()
        d_target_label_encoded = torch.full((batch_size, 1), 0, device=self.device).float()
        g_target_label_encoded = torch.full((batch_size, 1), 1, device=self.device).float()

        d_on_cover = self.discriminator(clean_images)
        d_loss_on_cover = self.BCE_loss_fn(d_on_cover, d_target_label_cover)
        self.optimizer_dis.zero_grad()
        d_loss_on_cover.backward()
        self.optimizer_dis.step()

        d_on_encoded = self.discriminator(encoded_images.detach())
        d_loss_on_encoded = self.BCE_loss_fn(d_on_encoded, d_target_label_encoded)

        self.optimizer_dis.zero_grad()
        d_loss_on_encoded.backward()
        self.optimizer_dis.step()

        D_loss = d_loss_on_cover + d_loss_on_encoded

        
        # encoder,decoder
        self.optimizer_en.zero_grad()
        self.optimizer_de.zero_grad()

        d_on_encoded_for_enc = self.discriminator(encoded_images)
        g_loss_adv = self.BCE_loss_fn(d_on_encoded_for_enc, g_target_label_encoded)
        
        if not self.opt.train.vgg_loss:
            g_loss_enc = self.mse_loss(encoded_images, clean_images)
        else:
            vgg_on_cov = self.vgg_loss(clean_images)
            vgg_on_enc = self.vgg_loss(encoded_images)
            g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)

        g_loss_dec = self.mse_loss(decoded_messages, messages)
        g_loss = self.opt.loss.adversarial_loss_weight * g_loss_adv + self.opt.loss.encoder_loss_weight * g_loss_enc \
                    + self.opt.loss.decoder_loss_weight * g_loss_dec
        
        
        g_loss.backward()
        self.optimizer_de.step()
        self.optimizer_en.step()

        # mi_loss
        if self.opt.train.mi_loss:
            for j in range(1):
                # forward
                encoded_images,_ = self.encoder(clean_images,messages)
                residual = encoded_images - clean_images
                with torch.no_grad():
                    decoder_output,_ = self.decoder(encoded_images,clean_images.clone())
                if self.opt.train.mi_loss_type == 'vsd':
                    residual_output,_ = self.decoder(residual,clean_images.clone())
                    po_kl_value = self.mi_fn((torch.sigmoid(residual_output)/self.opt.train.temperature).softmax(-1).log(),
                                            (torch.sigmoid(decoder_output)/self.opt.train.temperature).softmax(-1))
                    ne_kl_value = self.mi_fn((torch.sigmoid(decoder_output)/self.opt.train.temperature).softmax(-1).log(),
                                            (torch.sigmoid(residual_output)/self.opt.train.temperature).softmax(-1))
                    # ne_kl_value = 0
                    kl_value = po_kl_value - ne_kl_value
                    # print("kl_value:",kl_value)
                    mi_loss = kl_value*(-self.mi_loss_weight)
                    self.accelerator.wait_for_everyone()
                    self.optimizer_en.zero_grad()

                    mi_loss.backward()
                    self.optimizer_en.step()


        # metrics
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_accuracy = 1 - np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])
        psnr1, psnr2 ,ssim = psnr_ssim(clean_images,encoded_images,noised_images)


        # log # scale mi loss by multiply 1e6
        loss_list = [D_loss.item(),g_loss.item(),mi_loss.item()]
        metric_list = [bitwise_accuracy.item(),psnr1.item(),psnr2.item(),ssim.item()]
        
        image_dict = {
            "clean_image": make_grid(clean_images[0:4,:,:,:].detach().cpu()),
            "residual": make_grid(residual[0:4,:,:,:].abs().detach().cpu()),
            "image_with_watermark": make_grid(encoded_images[0:4,:,:,:].detach().cpu()),
            "noised_images": make_grid(noised_images[0:4,:,:,:].detach().cpu()),
        }
        return loss_list,metric_list,image_dict