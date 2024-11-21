import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random

def plot_samples(samples, h=5, w=10):
    plt.ioff()
    fig, axes = plt.subplots(
        nrows=h, ncols=w, figsize=(int(1.4 * w), int(1.4 * h)),
        subplot_kw={'xticks': [], 'yticks': []})
    for i, ax in enumerate(axes.flatten()):
        if 28 in samples[i].shape:#MNIST plotting
            ax.imshow(samples[i].squeeze(), cmap='gray')
        else:
            ax.imshow(samples[i].clip(0, 1))
    plt.close(fig)
    return fig


def tensorboard_add_samples(model, test_loader, aug_method, device):
    # load one batch from testset
    data, _ = next(iter(test_loader))
    data = data.to(device)

    # generate augmented samples
    if aug_method == 'nominal':
        defomred_samples = data
    elif aug_method == 'gaussianFull':
        defomred_samples = model._deformImageGaussianFull(data)
    elif aug_method == 'rotation':
        defomred_samples = model._GenImageRotation(data)
    elif aug_method == 'translation':
        defomred_samples = model._GenImageTranslation(data)
    elif aug_method == 'affine':
        defomred_samples = model._GenImageAffine(data)
    elif aug_method == 'scaling_uniform':
        defomred_samples = model._GenImageScalingUniform(data)
    elif aug_method == 'DCT':
        defomred_samples = model._GenImageDCT(data)
    else:
        raise Exception('Undefined Augmentation Method')

    defomred_samples = defomred_samples.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze()

    fig_clean = plot_samples(data.detach().cpu().numpy().transpose(0, 2, 3, 1))
    fig_corrupted = plot_samples(defomred_samples)

    return fig_clean, fig_corrupted

def set_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def str2msg(str):
    return [True if el=='1' else False for el in str]

def msg2str(msg):
    return "".join([('1' if el else '0') for el in msg])

def update_counts(dictionary, string):
    if string in dictionary.keys():
        dictionary[string] += 1
    else:
        dictionary[string] = 1
    return dictionary

def merge_dicts(dict1, dict2):
    merged_dict = dict1.copy() 

    for key, value in dict2.items():
        if key in merged_dict:
            merged_dict[key] += value
        else:
            merged_dict[key] = value

    return merged_dict

def get_sim_sum(dictionary, input_string,tolerant):
    weight_sum = 0

    for key in dictionary:
        if hamming_distance(key, input_string) <= tolerant:
            weight_sum += dictionary[key]

    return weight_sum

def hamming_distance(str1, str2):
    if str1 == -1 or str2 == -1:
        return 1e6
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

def get_top_two_keys(dictionary):
    if not dictionary:
        return None
    sorted_keys = sorted(dictionary, key=dictionary.get, reverse=True)
    return sorted_keys[:2]

def generate_random_messages(message_length, batch_size=1):
    '''
    return a tensor with dimension of (b,fs) and whose elements are randomly generated 0 or 1
    '''
    z = torch.zeros((batch_size, message_length), dtype=torch.float).random_(0, 2)
    return z

def create_file_path(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    open(file_path, 'a').close()
    return file_path

def cal_tolerant(total_bits, beta=0.05):
    from scipy.stats import binom
    for k in range(total_bits + 1):
        prob = 1 - binom.cdf(k - 1, total_bits, 0.5)  # 计算概率
        if prob <= beta:
            break
    return total_bits - k

def calculate_most_common_bit_string(bit_count_dict):
    bit_length = 100  # 01bit串的长度
    zero_counts = [0] * bit_length
    one_counts = [0] * bit_length
    
    # 统计每个位上0和1的出现次数
    for bit_string, count in bit_count_dict.items():
        for i in range(bit_length):
            if bit_string[i] == '0':
                zero_counts[i] += count
            else:
                one_counts[i] += count
    
    # 根据每个位上0和1的出现次数，决定该位的最终值是0还是1
    result_bits = []
    for i in range(bit_length):
        if zero_counts[i] > one_counts[i]:
            result_bits.append('0')
        else:
            result_bits.append('1')
    
    # 将结果列表转换为字符串
    result_bit_string = ''.join(result_bits)
    return result_bit_string


if __name__ == "__main__":
    print(cal_tolerant(40))