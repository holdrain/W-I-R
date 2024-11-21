# the metrics for watermark schemes
import kornia
import torch
import numpy as np

# psnr
def psnr_ssim(image, watermarking_img, noised_img):
    '''
    input: image tensor should be in [0,1]
    '''

    psnr_wm2co = kornia.metrics.psnr(
        image.detach().clamp(0, 1),
        watermarking_img.detach().clamp(0, 1),
        1,
    )
    if watermarking_img.size() != noised_img.size():
        psnr_wm2no = torch.zeros((1,))
    else:
        psnr_wm2no = kornia.metrics.psnr(
            image.detach().clamp(0, 1),
            noised_img.detach().clamp(0, 1),
            1,
        )
    # ssim
    ssim_wm2co = kornia.metrics.ssim(
        image.detach().clamp(0, 1),
        watermarking_img.detach().clamp(0, 1),
        window_size=11,
    ).mean()
    
    return psnr_wm2co, psnr_wm2no, ssim_wm2co

class BatchMetricsAccumulator:
    def __init__(self,model):
        self.model = model
        self.reset()
    def update(self, loss_list, metric_list):
        self.loss_list += np.array(loss_list)
        self.metric_list += np.array(metric_list)
        self.count += 1
    def compute_epoch_metric(self,method):
        if method == "mean":
            self.loss_dict = {key: round(value.item()/self.count,4) for key, value in zip(self.loss_dict.keys(), self.loss_list)}
            self.metric_dict = {key: round(value.item()/self.count,4) for key, value in zip(self.metric_dict.keys(), self.metric_list)}
        else:
            self.loss_dict = {key: round(value.item(),4) for key, value in zip(self.loss_dict.keys(), self.loss_list)}
            self.metric_dict = {key: round(value.item(),4) for key, value in zip(self.metric_dict.keys(), self.metric_list)}

        return self.loss_dict,self.metric_dict

    def reset(self):
        if self.model == "stega":
            self.loss_dict = {
                "loss/total_loss":0.0,
                "loss/D_loss":0.0,
                "loss/G_loss":0.0,
                "loss/l2_loss":0.0,
                "loss/lpip_loss":0.0,
                "loss/mi_loss":0.0,
                "loss/BCE_loss":0.0,
            }
        else:
            self.loss_dict = {
                "loss/D_loss":0.0,
                "loss/g_loss":0.0,
                "loss/mi_loss":0.0,
            }
        
        self.metric_dict = {
            'bit-acc':0.0,
            'PSNR(W-C)':0.0,
            'PSNR(N-W)':0.0,
            'SSIM(W-C)':0.0,
        }
        self.count = 0
        self.loss_list = np.zeros((len(self.loss_dict.keys()),))
        self.metric_list = np.zeros((len(self.metric_dict.keys()),))
        
        
    