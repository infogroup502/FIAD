import argparse
import os
import sys
import time
import warnings
import numpy as np
import torch
from torch.backends import cudnn
from solver import Solver
from utils.utils import *
warnings.filterwarnings('ignore')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def list_type(arg):
    return [int(item) for item in str(arg).split(',')]


def main(config):
    cudnn.benchmark = True
    if not os.path.exists(config.model_save_path):
        mkdir(config.model_save_path)
    solver = Solver(vars(config))
    if config.mode == 'train':
        solver.run()
    if config.mode == 'test':
        solver.test()
    return solver


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--win_size', type=int, default=10)
    parser.add_argument('--time_size', type=list_type, default=[10])
    parser.add_argument('--spatial_size', type=list_type, default=[8])
    parser.add_argument('--anormly_ratio', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--r', type=float, default=0.5)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--step', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=True)
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    parser.add_argument('--loss_fuc', type=str, default='MSE')
    parser.add_argument('--index', type=int, default=137)
    parser.add_argument('--input_c', type=int, default=18)
    parser.add_argument('--output_c', type=int, default=18)
    parser.add_argument('--dataset', type=str, default='Genesis')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='Genesis')
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--stage1_lr', type=float, default=1e-3)
    parser.add_argument('--stage2_lr', type=float, default=1e-4)
    parser.add_argument('--time_num_sin_waves', type=int, default=20)
    parser.add_argument('--spatial_num_sin_waves', type=int, default=20)
    config = parser.parse_args()
    set_seed(42)
    config.use_gpu = True if torch.cuda.is_available() and config.use_gpu else False

    if config.use_gpu and config.use_multi_gpu:
        config.devices = config.devices.replace(' ', '')
        device_ids = config.devices.split(',')
        config.device_ids = [int(id_) for id_ in device_ids]
        config.gpu = config.device_ids[0]

    if not os.path.exists(config.model_save_path):
        mkdir(config.model_save_path)

    class Logger(object):
        def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
            self.terminal = stream
            self.filename = filename
            self.add_flag = add_flag
        def write(self, message):
            if self.add_flag:
                with open(self.filename, 'a+', encoding='utf-8') as log:
                    self.terminal.write(message)
                    log.write(message)
            else:
                with open(self.filename, 'w', encoding='utf-8') as log:
                    self.terminal.write(message)
                    log.write(message)
        def flush(self):
            pass
    log_dir = "all_resules/result_FIAD"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    sys.stdout = Logger(log_dir + "/" + config.data_path + ".log", stream=sys.stdout)

    if config.mode == 'train':
        print("\n\n")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print('================ Hyperparameters ===============')
        for k, v in sorted(vars(config).items()):
            print('%s: %s' % (str(k), str(v)))
        print('====================  Train  ===================')

    main(config)