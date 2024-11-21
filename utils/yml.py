import yaml
from yaml import Loader, Dumper
from collections import OrderedDict
import random
import numpy as np
import torch

def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def parse_yml(opt_path):
    Loader, Dumper = OrderedYaml()
    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=Loader)
    return opt

class NoneDict(dict):
    def __missing__(self, key):
        return None
# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt
    
# update param dict
def update_param(opt1,opt2):
    opt1.data.dataset = opt2.dataset if opt2.dataset is not None else opt1.data.dataset
    opt1.data.num = opt2.num if opt2.num is not None else opt1.data.num
    opt1.noise.choice = opt2.noise_choice if opt2.noise_choice is not None else opt1.noise.choice
    opt1.noise.sigma = opt2.sigma if opt2.sigma is not None else opt1.noise.sigma
    opt1.train.batch_size = opt2.batch_size if opt2.batch_size is not None else opt1.train.batch_size
    opt1.val.batch_size = opt2.batch_size if opt2.batch_size is not None else opt1.val.batch_size
    opt1.resume_path = opt2.resume_path if opt2.resume_path is not None else opt1.resume_path
    opt1.train.epoch = opt2.epoch if opt2.epoch is not None else opt1.train.epoch
    opt1.train.set_start_epoch = opt2.set_start_epoch if opt2.set_start_epoch is not None else opt1.train.set_start_epoch
    if hasattr(opt1,"lre"):
        opt1.lre = opt2.lre if opt2.lre is not None else opt1.lre
        opt1.lrd = opt2.lrd if opt2.lrd is not None else opt1.lrd
    else:
        opt1.lr = opt2.lr if opt2.lr is not None else opt1.lr
    opt1.train.optimizer = opt2.optimizer if opt2.optimizer is not None else opt1.train.optimizer
    print(opt1.data.dataset)
    return opt1
