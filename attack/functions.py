
import sys
sys.path.append("/mnt/shared/Huggingface/sharedcode/Stegastamp_CR")
sys.path.append("/data/shared/Huggingface/sharedcode/Stegastamp_CR")
from utils.helpers import *
import torch
from PIL import Image
import matplotlib.pyplot as plt
import kornia

def extract_message(decoder,image_tensor,device,model_choice):
    decoder.to(device)
    decoder.eval()
    decoder_output = decoder(image_tensor)
    with torch.inference_mode():
        if model_choice == "stega":
            fingerprint = (decoder_output>0).long()
        else:
            fingerprint = (decoder_output.round().clip(0, 1)>0.5).long()

    return fingerprint,decoder_output

def get_wmimages(ds,image_i,text,encoder,device,model_choice):
    '''
    return a tensor of watermarked images and residual images
    image_i: a sequence of original images in dataset
    '''
    fingerprints = text.to(device)
    encoder = encoder.to(device)
    encoder = encoder.eval()
    fingerprinted_images = []
    residual_images = []
    # loop
    with torch.inference_mode():
        for idx in image_i:
            image = ds[idx][0].unsqueeze(0).to(device)
            if model_choice == "stega":
                fingerprint_image,_,residual_image = encoder(fingerprints,image)
            else:
                fingerprint_image,_ = encoder(image,fingerprints)
                residual_image = fingerprint_image - image
            residual_images.append(residual_image)
            fingerprinted_images.append(fingerprint_image)

    return torch.cat(fingerprinted_images,dim=0),torch.cat(residual_images,dim=0)

def get_images(ds,image_i,device):
    '''
    return a list of clean images in index list image_i
    '''
    images = []
    # loop
    for idx in image_i:
        image = ds[idx][0].to(device)
        images.append(image)
    return torch.stack(images,dim=0)
    
def show(image,title):
    if image.device.type == "cuda":
        image = image.detach().cpu()
    if image.dim() == 4:
        image = image.squeeze(0)
    plt.imshow(image.permute(1,2,0))
    plt.title(title)
    plt.axis('off')
    plt.show()


def calculate_similarity_sigle(t1,t2):
    similarity = torch.nn.functional.cosine_similarity(torch.flatten(t1),torch.flatten(t2),dim=0)
    return similarity

def calculate_similarity(tensors):
    n_tensors = len(tensors)
    similarity_matrix = np.zeros((n_tensors, n_tensors))
    s_sum = 0.0
    for i in range(n_tensors):
        for j in range(n_tensors):
            similarity = calculate_similarity_sigle(tensors[i],tensors[j])
            similarity_matrix[i, j] = similarity.detach().cpu().numpy()
            s_sum += similarity_matrix[i, j]
    return similarity_matrix,s_sum / (n_tensors*n_tensors)


def hamming_distance(str1, str2):
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

def t_distance(t0,text_set):
    distances = [hamming_distance(t0, t) for t in text_set]
    return distances



def generate_random_fingerprints(fingerprint_size, batch_size=4):
    '''
    return a tensor with dimension of (b,fs) and whose elements are randomly generated 0 or 1
    '''
    z = torch.zeros((batch_size, fingerprint_size), dtype=torch.float).random_(0, 2)
    return z


def get_single_image(image_path,tranform):
    image = Image.open(image_path).convert("RGB")
    image_tensor = tranform(image).unsqueeze(0)
    return image_tensor


from attacks.attack import WMAttacker
def get_residual_prediction(wm_images,batch_size,device,method):
    '''
    the function is used for predicting residual images of watermarked images
    wm_images: the watermakered images tensor with shape of (N,3,h,w)
    method: can be VAE or low pass
    '''
    if method == 'VAE':
        wmattacker = WMAttacker(
            attack_list=["regen_vae"],
            strength = 6,
            vae_method = "sdxl_vae",
            image_size=128,
            device=device,
            transform=None,
            batch_size=None,
        )
    elif method == 'diffusion':
        wmattacker = WMAttacker(
            attack_list=["regen_diffusion"],
            vae_method = "",
            strength = 2,
            image_size=128,
            device=device,
            transform=None,
            batch_size=None,
        )
    N = wm_images.size(0)
    results = []
    for i in range(0,N,batch_size):
        interval = min(batch_size,N-i)
        batch_data = wm_images[i:i+interval,:,:,:]
        res = wmattacker.attack_batch(batch_data)
        # print(f"batch_size:{batch_data.size()},res.size:{res.size()}")
        batch_residual = batch_data - res
        
        results.append(batch_residual)
    return torch.cat(results,dim=0)

def calculate_mean_and_variance(data):
    """Calculate the mean and variance of a list of numbers."""
    if len(data) == 0:
        raise ValueError("The data list is empty")
    
    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    
    return mean, variance

def psnr_ssim(image, watermarking_img, noised_img):
    '''
    input: image tensor should be in [0,1]
    '''
    psnr_wm2co = kornia.metrics.psnr(
        image.detach().clamp(0, 1),
        watermarking_img.detach().clamp(0, 1),
        1,
    )
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

name_dict = {
    'OO':"Clean",
    'emperical':"Empirical",
    'GN_0.1':"Gaussian-0.10",
    'GN_0.25':"Gaussian-0.25",
    'GN_0.5':"Gaussian-0.50",
    'affine_0.01':"Affine-0.01",
    'affine_0.02':"Affine-0.02",
    'affine_0.03':"Affine-0.03",
 }



if __name__ == '__main__':
    print(generate_random_fingerprints(100,1).shape)