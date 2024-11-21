import math
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import binomtest, norm
from statsmodels.stats.proportion import proportion_confint

from utils.certifyutils import *


class WMSmooth(object):
    """A smoothed decoder """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self,model:str, base_decoder: torch.nn.Module, tolerant: int, sigma: float, certify_method: str,device: torch.device):
        """
        :param base_decoder: maps from [batch x channel x height x width] to [batch x message_length]
        :param sigma: the noise level hyperparameter
        :param tolerant: a threshold of certification of watermark scheme which is relervant to message_length
        :certify_method:
        """
        self.device = device
        self.model_choice = model
        self.base_decoder = base_decoder
        self.sigma = sigma
        self.tolerant = tolerant
        self.certify_method = certify_method
        if self.certify_method == 'rotation':
            self.sigma *= math.pi
        
    def _GenDeformGaussian(self, imgs, N, device):
        ''' This function takes an image C x W x H and returns N Gaussianly perturbed coordinates versions'''       
        batch = imgs.repeat((N, 1, 1, 1))
        num_channels, rows, cols = imgs.shape
        randomFlow = torch.randn(N, rows, cols, 2, device=device) * self.sigma

        new_ros = torch.linspace(-1, 1, rows)
        new_cols = torch.linspace(-1, 1, cols)

        meshx, meshy = torch.meshgrid((new_ros, new_cols))
        grid = torch.stack((meshy, meshx), 2).unsqueeze(0).expand(N, rows, cols, 2).to(device)

        new_grid = grid + randomFlow

        Iwarp = F.grid_sample(batch, new_grid, align_corners=True)
        return Iwarp


    def _GenImageRotation(self, x, N ,device):
        _, rows, cols = x.shape  #Usually in certification, the batch size is 1
        ang = (-2 * torch.rand((N, 1, 1)) + 1) *self.sigma #Uniform between [-sigma, sigma]
        
        #Generating the vector field for rotation. Not that sigma should be sig*pi, where sig is in [0,1]
        X, Y = torch.meshgrid(torch.linspace(-1,1,rows),torch.linspace(-1,1,cols))
        X, Y = X.unsqueeze(0), Y.unsqueeze(0)
        Xv = X*torch.cos(ang)-Y*torch.sin(ang)-X
        Yv = X*torch.sin(ang)+Y*torch.cos(ang)-Y
        
        randomFlow = torch.stack((Yv,Xv), axis=3).to(device)
        grid = torch.stack((Y,X), axis=3).to(device)
        
        return F.grid_sample(x.repeat((N, 1, 1, 1)), grid+randomFlow,align_corners=True)

    def _GenImageTranslation(self, x, N, device):
        _, rows, cols = x.shape #N is the batch size

        #Generating the vector field for translation.
        X, Y = torch.meshgrid(torch.linspace(-1,1,rows),torch.linspace(-1,1,cols))
        X, Y = X.unsqueeze(0), Y.unsqueeze(0)
        Xv = torch.randn((N, 1, 1))*self.sigma + 0*X
        Yv = torch.randn((N, 1, 1))*self.sigma + 0*Y
        
        randomFlow = torch.stack((Yv,Xv), axis=3).to(device)
        grid = torch.stack((Y,X), axis=3).to(device)
        
        return F.grid_sample(x.repeat((N, 1, 1, 1)), grid+randomFlow,align_corners=True)
    
    def _GenImageScalingUniform(self, x, N, device):
        _, rows, cols = x.shape # N is the batch size
        #Scaling here is sampled from uniform distribution between [1-sigma, 1+sigma]
        scale = (-2 * torch.rand((N, 1, 1)) + 1.0) * self.sigma + 1.0
        #Generating the vector field for scaling.
        X, Y = torch.meshgrid(torch.linspace(-1,1,rows),torch.linspace(-1,1,cols))
        X, Y = X.unsqueeze(0), Y.unsqueeze(0)
        Xv = X * scale - X
        Yv = Y * scale - Y
        
        randomFlow = torch.stack((Yv,Xv), axis=3).to(device)
        grid = torch.stack((Y,X), axis=3).to(device)
        
        return F.grid_sample(x.repeat((N, 1, 1, 1)), grid+randomFlow,align_corners=True)

    def _GenImageAffine(self, x, N, device):
        _, rows, cols = x.shape # N is the batch size
        
        params = torch.randn((6, N, 1, 1))*self.sigma

        #Generating the vector field for Affine transformation.
        X, Y = torch.meshgrid(torch.linspace(-1,1,rows),torch.linspace(-1,1,cols))
        X, Y = X.unsqueeze(0), Y.unsqueeze(0)
        Xv = params[0]*X + params[1]*Y + params[2]
        Yv = params[3]*X + params[4]*Y + params[5]
        
        randomFlow = torch.stack((Yv,Xv), axis=3).to(device)
        grid = torch.stack((Y,X), axis=3).to(device)
        
        return F.grid_sample(x.repeat((N, 1, 1, 1)), grid+randomFlow,align_corners=True)

    def _GenImageDCT(self, x, N, device):

        _, rows, cols = x.shape
        new_ros = torch.linspace(-1, 1, rows)
        new_cols = torch.linspace(-1, 1, cols)
        meshx, meshy = torch.meshgrid((new_ros, new_cols))
        grid = torch.stack((meshy, meshx), 2).unsqueeze(0).expand(N, rows, cols, 2).to(self.device)

        X, Y = torch.meshgrid((new_ros, new_cols))
        X = torch.reshape(X, (1, 1, 1, rows, cols))
        Y = torch.reshape(Y, (1, 1, 1, rows, cols))

        param_ab = torch.randn(N, self.num_bases, self.num_bases, 1, 2) * self.sigma
        a = param_ab[:, :, :, :, 0].unsqueeze(4)
        b = param_ab[:, :, :, :, 1].unsqueeze(4)
        K1 = torch.arange(self.num_bases).view(1, self.num_bases, 1, 1, 1)
        K2 = torch.arange(self.num_bases).view(1, 1, self.num_bases, 1, 1)
        basis_factors  = torch.cos(math.pi* (K1 * (X+0.5/rows) ))*torch.cos(math.pi * (K2 * (Y+0.5/cols)))

        U = torch.squeeze(torch.sum(a * basis_factors, dim=(1, 2)))
        V = torch.squeeze(torch.sum(b * basis_factors, dim=(1, 2)))

        randomFlow = torch.stack((V, U), dim=3).to(device)

        return F.grid_sample(x.repeat((N, 1, 1, 1)), grid + randomFlow,align_corners=True)
    
    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int, label: str) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_decoder.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size,torch.tensor(str2msg(label)))
        # use these samples to take a guess at the top class
        # cAHat = get_top_two_keys(counts_selection)[0]
        # # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size,torch.tensor(str2msg(label)))
        # # print(cAHat,label)
        # error_bits = int(hamming_distance(cAHat,label))
        # if error_bits <= self.tolerant:
            
        #     nA = get_sim_sum(counts_estimation,label,self.tolerant)
        # else:
        #     nA = get_sim_sum(counts_estimation,cAHat,self.tolerant)
        # pABar = self._lower_confidence_bound(nA, n, alpha)
        # # print(f"nA:{nA},pABar:{pABar},n:{n}")
        # if pABar < 0.5:
        #     return WMSmooth.ABSTAIN, 0.0 ,0.5,error_bits
        # else:
        #     radius = self.sigma * norm.ppf(pABar)
        #     return cAHat, radius ,pABar,error_bits
        
        ### scheme 1
        # max_nA = 0
        # cAHat = ''
        # for ca in counts_selection.keys():
        #     nA = get_sim_sum(counts_estimation,ca,self.tolerant)
        #     if nA > max_nA:
        #         cAHat = ca
        #         max_nA = nA
        # pABar = self._lower_confidence_bound(nA, n, alpha)
        # # print(f"nA:{nA},pABar:{pABar},n:{n}")
        # error_bits = int(hamming_distance(cAHat,label))
        # if pABar < 0.5:
        #     return WMSmooth.ABSTAIN, 0.0 ,0.5,error_bits
        # else:
        #     radius = self.sigma * norm.ppf(pABar)
        #     return cAHat, radius ,pABar,error_bits
        

        ### scheme 2
        # cAHat = calculate_most_common_bit_string(counts_selection)
        
        # nA = get_sim_sum(counts_estimation,cAHat,self.tolerant)
            
        # pABar = self._lower_confidence_bound(nA, n, alpha)
        # # print(f"nA:{nA},pABar:{pABar},n:{n}")
        # error_bits = int(hamming_distance(cAHat,label))
        # if pABar < 0.5:
        #     return WMSmooth.ABSTAIN, 0.0 ,0.5,error_bits
        # else:
        #     radius = self.sigma * norm.ppf(pABar)
        #     return cAHat, radius ,pABar,error_bits
                

        ### scheme 3
        # cAHat = calculate_most_common_bit_string(counts_selection)
        
        # nA = get_sim_sum(counts_estimation,label,self.tolerant)
            
        # pABar = self._lower_confidence_bound(nA, n, alpha)
        # # print(f"nA:{nA},pABar:{pABar},n:{n}")
        # error_bits = int(hamming_distance(cAHat,label))
        # if pABar < 0.5:
        #     return WMSmooth.ABSTAIN, 0.0 ,0.5,error_bits
        # else:
        #     radius = self.sigma * norm.ppf(pABar)
        #     return cAHat, radius ,pABar,error_bits

        # scheme ZP
        cAHat = counts_selection.argmax().item()
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return WMSmooth.ABSTAIN, 0.0, 0.5#0.5 for the radius to be zero
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius, pABar
        

        # use these samples to estimate a lower bound on pA
        

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_decoder.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = get_top_two_keys(counts)
        count1 = counts[top2[0]]
        if len(top2) == 2:
            count2 = counts[top2[1]]
        else:
            count2 = count1
        if binomtest(count1, count1 + count2, p=0.5).pvalue > alpha:
            return WMSmooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size,label) -> dict:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size: num/batch_size batches will be generated and the image tensors in a batch shared a noise  
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(2, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size
                if self.certify_method == 'OO':
                    pass
                elif self.certify_method == 'gaussianFull':
                    batch = self._GenDeformGaussian(x, this_batch_size, device=self.device)
                elif self.certify_method == 'rotation':
                    batch = self._GenImageRotation(x, this_batch_size, device=self.device)
                elif self.certify_method == 'translation':
                    batch = self._GenImageTranslation(x, this_batch_size, device=self.device)
                elif self.certify_method == 'affine':
                    batch = self._GenImageAffine(x, this_batch_size, device=self.device)
                elif self.certify_method == 'scaling_uniform':
                    batch = self._GenImageScalingUniform(x, this_batch_size, device=self.device)
                elif self.certify_method == 'DCT':
                    batch = self._GenImageDCT(x, this_batch_size, device=self.device)
                elif self.certify_method == 'GN':
                    batch = x + torch.randn_like(x.repeat((this_batch_size, 1, 1, 1))) * self.sigma
                else:
                    raise Exception('Undefined augmentaion method!')

                decoder_output = self.base_decoder(batch)
                if self.model_choice == 'stega':
                    bits_predictions = (decoder_output>0).long()
                else:
                    bits_predictions = (decoder_output.round().clip(0, 1)>0.5).long()
                # a array recording the num fo each predicted class

                # scheme
                # counts = merge_dicts(counts,self._count_arr(predictions.cpu().numpy()))

                # scheme zp
                label = label.to(self.device)
                matches = (bits_predictions == label.repeat(this_batch_size,1))
                
                # 计算匹配位数
                matches_count = matches.sum(dim=1)
                # print(matches_count)
                # print(matches.shape)
                class_predictions = (matches_count > 58).int()
                # print(class_predictions[0:10])
                counts += self._count_arr_zp(class_predictions,2)

            return counts

    def _count_arr(self, arr: np.ndarray) -> dict:
        counts = {}
        for index in range(arr.shape[0]):
            message = arr[index,:].tolist()
            message = msg2str(message)
            update_counts(counts,message)
        return counts
    
    def _count_arr_zp(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]