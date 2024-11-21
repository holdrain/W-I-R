import torch
from torch import nn
import wandb
from easydict import EasyDict
from utils.metrics import psnr_ssim
from torchvision.utils import make_grid
from utils.AutoWeightedLoss import AutomaticWeightedLoss
from models.mi_estimator import CLUBForCategorical


class Trainer():
    def __init__(self,opt,optimizer_en,optimizer_de,optimizer_dis,encoder,decoder,discriminator,accelerator,lpips_net) -> None:
        # setting
        self.opt = opt
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator
        self.optimizer_dis = optimizer_dis
        self.optimizer_en = optimizer_en
        self.optimizer_de = optimizer_de
        # fn
        self.l2_loss_fn = nn.MSELoss()
        self.lpips_fn = lpips_net
        self.BCE_loss_fn = nn.BCEWithLogitsLoss()

        if opt.train.mi_loss_type == 'kl' or opt.train.mi_loss_type == 'vsd':
            self.mi_fn = nn.KLDivLoss(reduction='batchmean')
            self.mi_loss_weight = opt.loss.mi_loss_weight
        elif opt.train.mi_loss_type == 'club':
            self.mi_fn = CLUBForCategorical(decoder=self.decoder)
            self.mi_loss_weight = opt.loss.mi_loss_weight
        else:
            self.mi_loss_weight = 0
        
        if self.opt.train.autoweighted:
            self.awl = AutomaticWeightedLoss(3)
        self.accelerator = accelerator
        
    def train_on_batch(self,clean_images,messages,activated_step):
        '''
        out_dict: {clean images, watermarked images, tup, message_prediction,D_output_real,D_output_fake}
        message: gt
        '''
        self.encoder.train()
        self.decoder.train()
        self.discriminator.train()


        batch_size = clean_images.size(0)
        watermarked_images,tup,_ = self.encoder(messages, clean_images)
        residual = watermarked_images - clean_images
        decoder_output,noised_images = self.decoder(watermarked_images)

        # Train Discriminator
        D_loss = torch.tensor(0)
        if self.opt.train.gan_loss and activated_step > self.opt.train.no_im_loss_steps:
            self.optimizer_dis.zero_grad()
            D_output_real, _ = self.discriminator(clean_images)
            D_output_fake, _ = self.discriminator(watermarked_images.detach())
            D_loss = D_output_real - D_output_fake
            D_loss.backward()
            self.optimizer_dis.step()
        
        # Train encoder decoder
        self.optimizer_en.zero_grad()
        self.optimizer_de.zero_grad()
        l2_loss_weight = min(self.opt.loss.l2_loss_weight * activated_step / self.opt.loss.l2_loss_ramp, self.opt.loss.l2_loss_weight)
        lpips_loss_weight = min(self.opt.loss.lpips_loss_weight * activated_step / self.opt.loss.lpips_loss_ramp, self.opt.loss.lpips_loss_weight)
        G_loss_weight = min(self.opt.loss.G_loss_weight * activated_step / self.opt.loss.G_loss_ramp, self.opt.loss.G_loss_weight)

        # loss
        loss = torch.tensor(0)
        G_loss = torch.tensor(0)
        mi_loss = torch.tensor(0)
        l2_loss = torch.tensor(0)
        lpips_loss = torch.tensor(0)
        image_loss = torch.tensor(0)

        if activated_step > self.opt.train.no_im_loss_steps:
            # image loss
            l2_loss = self.l2_loss_fn(watermarked_images, clean_images)
            normalized_input,normalized_encoded = clean_images * 2 - 1, watermarked_images * 2 -1
            lpips_loss = torch.mean(self.lpips_fn(normalized_input, normalized_encoded))
            image_loss = l2_loss_weight * l2_loss + lpips_loss_weight * lpips_loss
            # generator loss
            if self.opt.train.gan_loss:
                D_output_fake, _ = self.discriminator(watermarked_images)
                G_loss = D_output_fake

        # only bce loss
        BCE_loss = self.BCE_loss_fn(decoder_output.view(-1),messages.view(-1))
        if self.opt.train.autoweighted:
            loss = self.awl([image_loss,BCE_loss,G_loss])
        else:
            loss = image_loss + self.opt.loss.BCE_loss_weight * BCE_loss + G_loss * G_loss_weight
        loss.backward()
        self.optimizer_de.step()
        self.optimizer_en.step()


        # mi loss
        for j in range(5):
            if self.opt.train.mi_loss:
                # forward
                watermarked_images,tup,_ = self.encoder(messages, clean_images)
                residual = watermarked_images - clean_images
                with torch.no_grad():
                    decoder_output,_ = self.decoder(watermarked_images)
                if self.opt.train.mi_loss_type == 'kl':
                    mi_loss = -1 * self.mi_fn(residual.view(batch_size,-1).softmax(-1).log(),
                                    target=tup.view(batch_size,-1).softmax(-1)) * self.mi_loss_weight
                    # mi loss grad only on encoder
                    self.optimizer_en.zero_grad()
                    mi_loss.backward()
                    self.optimizer_en.step()
                elif self.opt.train.mi_loss_type == 'vsd':
                    residual_output,_ = self.decoder(residual)
                    po_kl_value = self.mi_fn((torch.sigmoid(residual_output)/self.opt.train.temperature).softmax(-1).log(),
                                            (torch.sigmoid(decoder_output)/self.opt.train.temperature).softmax(-1))
                    ne_kl_value = self.mi_fn((torch.sigmoid(decoder_output)/self.opt.train.temperature).softmax(-1).log(),
                                            (torch.sigmoid(residual_output)/self.opt.train.temperature).softmax(-1))
                    # ne_kl_value = 0
                    kl_value = po_kl_value - ne_kl_value
                    mi_loss = kl_value*(-self.mi_loss_weight)
                    self.optimizer_en.zero_grad()
                    mi_loss.backward()
                    self.optimizer_en.step()
                elif self.opt.train.mi_loss_type == 'club':
                    self.mi_fn.update_net(self.decoder)
                    self.mi_fn.eval()
                    mi_loss = self.mi_fn(residual,decoder_output) * self.mi_loss_weight
                    self.optimizer_en.zero_grad()
                    mi_loss.backward()
                    self.optimizer_en.step()
                    pass

        # metrics
        message_prediction = (decoder_output > 0).float()
        bitwise_accuracy = 1.0 - torch.mean(torch.abs(messages - message_prediction))
        psnr1, psnr2 ,ssim = psnr_ssim(clean_images,watermarked_images,noised_images)

        # log
        loss_list = [loss.item(),D_loss.item(),G_loss.item(),l2_loss.item(),lpips_loss.item(),mi_loss.item(),BCE_loss.item()]
        metric_list = [bitwise_accuracy.item(),psnr1.item(),psnr2.item(),ssim.item()]

        image_dict = {
            "clean_images": make_grid(clean_images[0:4,:,:,:].detach().cpu()),
            "residual": make_grid(residual[0:4,:,:,:].abs().detach().cpu()),
            "image_with_watermark": make_grid(watermarked_images[0:4,:,:,:].detach().cpu()),
            "noised_images": make_grid(noised_images[0:4,:,:,:].detach().cpu()),
            "upsamle_message":make_grid(tup[0:4,:,:,:].detach().cpu()),
        }
        return loss_list,metric_list,image_dict
