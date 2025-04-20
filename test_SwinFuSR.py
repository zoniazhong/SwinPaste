'''
# --------------------------------------------------------
# References:
# https://github.com/VisionICLab/SwinFuSR
# --------------------------------------------------------
'''
import os.path
import math
import argparse
import time
import random
import numpy as np
from collections import OrderedDict
import logging
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch
from torchsummary import summary

from utils import utils_logger
from utils import utils_image as util
from utils import utils_option as option
from utils.utils_dist import get_dist_info, init_dist
import wandb
import time
import cv2

from data.select_dataset import define_Dataset
from models.select_model import define_Model
import warnings
warnings.filterwarnings("ignore")


def main(json_path='/root/autodl-tmp/root/autodl-tmp/SwinFuSR-main/options/test_swinFuSR_x8.json'):
# def main(json_path='/root/autodl-tmp/root/autodl-tmp/SwinFuSR-main/options/test_swinFuSR_x16.json'):
    '''
    # ----------------------------------------
    # Step--1 (prepare opt)
    # ----------------------------------------
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=json_path, help='Path to option JSON file.')
    parser.add_argument('--launcher', default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', default=False)

    opt = option.parse(parser.parse_args().opt, is_train=True)
    opt['dist'] = parser.parse_args().dist

    # ----------------------------------------
    # distributed settings
    # ----------------------------------------
    if opt['dist']:
        init_dist('pytorch')
    opt['rank'], opt['world_size'] = get_dist_info()
    
    if opt['rank'] == 0:
        for key, path in opt['path'].items():
            print(path)
        util.mkdirs((path for key, path in opt['path'].items() if 'pretrained' not in key))

    # ----------------------------------------
    # update opt
    # ----------------------------------------
    # -->-->-->-->-->-->-->-->-->-->-->-->-->-
    init_iter_G, init_path_G = option.find_last_checkpoint(opt['path']['models'], net_type='G')
    init_iter_E, init_path_E = option.find_last_checkpoint(opt['path']['models'], net_type='E')

    init_iter_optimizerG, init_path_optimizerG = option.find_last_checkpoint(opt['path']['models'], net_type='optimizerG')
    opt['path']['pretrained_optimizerG'] = init_path_optimizerG
    current_step = max(init_iter_G, init_iter_E, init_iter_optimizerG)

    border = opt['scale']
    # --<--<--<--<--<--<--<--<--<--<--<--<--<-

    # ----------------------------------------
    # save opt to  a '../option.json' file
    # ----------------------------------------
    if opt['rank'] == 0:
        option.save(opt)

    # ----------------------------------------
    # return None for missing key
    # ----------------------------------------
    opt = option.dict_to_nonedict(opt)
    
    seed = 60
    if seed is None:
        seed = random.randint(1, 10000)
    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    '''
    # ----------------------------------------
    # Step--2 (creat dataloader)
    # ----------------------------------------
    '''

    # ----------------------------------------
    # 1) create_dataset
    # 2) creat_dataloader for test
    # ----------------------------------------
    
    for phase, dataset_opt in opt['datasets'].items():
        # if phase == 'validation':
        #     val_set = define_Dataset(dataset_opt)
        #     if opt['rank'] == 0:
        #         val_dataset = DataLoader(val_set, batch_size=1,
        #                              shuffle=dataset_opt['dataloader_shuffle'], num_workers=1,
        #                              drop_last=False, pin_memory=True)
        if phase == 'test':
            test_set = define_Dataset(dataset_opt)
            test_dataset = DataLoader(test_set, batch_size=1,
                                     shuffle=False, num_workers=1,
                                     drop_last=False, pin_memory=True)
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    '''
    # ----------------------------------------
    # Step--3 (initialize model)
    # ----------------------------------------
    '''

    model = define_Model(opt)
    model.load()  # load the model



    '''
    # ----------------------------------------
    # Step--4 (main test)
    # ----------------------------------------
    '''

    for _,test_set in enumerate(test_dataset):
            print('Processing ', test_set['Guide_path'])
            image_name_ext = os.path.basename(test_set['Guide_path'][0])
            save_img_path = os.path.join(opt['path']['images'], '{:s}'.format(image_name_ext))
            model.feed_data(test_set, phase='test', need_GT=False)
            model.test()
            visuals = model.current_visuals(need_H=False)
            E_img = util.tensor2uint(visuals['Output'])
            util.imsave(E_img, save_img_path)


if __name__ == '__main__':
    main()