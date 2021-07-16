import sys
sys.path.insert(0, '/home/dmitry/МАГА/Диплом/project/')

import json
import torch
from gtn.models.mnist_teacher import (MNISTTeacherConfig, MNISTTeacher,
                                      TeacherInputType)
from my_experiments.teacher import DDTeacher, DDTeacherConfig
from my_experiments.utils import (prepare_data, LoopConfig, TeacherIterDataset, 
                                  TeacherMapDataset, evaluate_net)
from pathlib import Path
import matplotlib.pyplot as plt
from gtn.datasets.datasets import Datasets
from gm.utils import get_network
from dataclasses import asdict
from collections import defaultdict
import os.path as osp
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
import time
import argparse

param_augment = {'crop': 4,
                 'scale': 0.2,
                 'rotate': 45,
                 'noise': 0.001,
                 'strategy': 'crop_scale_rotate',
                }


def parse_args():
    parser = argparse.ArgumentParser(description='Look for results')
    parser.add_argument('res_pth', type=str, help='pth to res file')
    parser.add_argument('dir', type=str, help='dir with tries')
    parser.add_argument('--is_dd', action='store_true', help='is Data Distillation')
    parser.add_argument('--do_aug', action='store_true', help='do augmentation')
    parser.add_argument('n_epochs', type=int, help='number of epochs')
    args = parser.parse_args()
    return args


def eval_step(testloader, loop_config, teacher, device, n_epochs=1000):
    st = time.time()
    learner = get_network(loop_config.learner_c.arch, 
                          **asdict(loop_config.data_c))
    teacher.eval()
    if loop_config.learned_curiculum:
        trainloader = TeacherMapDataset(teacher)
    else:
        trainloader = TeacherIterDataset(teacher)
    param_augment = loop_config.eval_c.aug_c
    res = evaluate_net(learner, trainloader, testloader, 
                       loop_config.eval_c.lr, param_augment, 
                       device, Epoch=n_epochs)
    _, acc_train, acc_test, loss_train = res
    return acc_train, acc_test, loss_train, time.time() - st


def load_config_and_model(dir_, is_dd:bool, do_aug:bool):
    # load config
    loop_config = LoopConfig.load(dir_/'loop_c.json')
    if do_aug: loop_config.eval_c.aug_c = param_augment
    
    # load teacher
    teacher_dir = Path(dir_/'teacher')
    with open(teacher_dir/'config.json') as cf: config = json.loads(cf.read())
    if is_dd:
        teacher = DDTeacher(DDTeacherConfig(**config), device='cpu').eval()
    else:
        config['input_type'] = TeacherInputType.from_string(config['input_type'])
        config = MNISTTeacherConfig(**config)
        teacher = MNISTTeacher(config, device='cpu').eval()
    
    # state_dict
    with open(teacher_dir/'model.bin', 'rb') as f: state = torch.load(f)
    teacher.load_state_dict(state)
    teacher.eval()
    return loop_config, teacher


if __name__ == '__main__':
    args = parse_args()
    res = {}
    if osp.exists(args.res_pth):
        with open(args.res_pth, 'r') as f:
            res = json.loads(f.read())
    test_loader = Datasets.mnist_dataloader(512, train=False)
    for learner_arch in ['LeNet', 'AlexNet', 'VGG11', 'MLP']:
        for try_ in range(3):
            dir_ = Path(args.dir)/f'try-{try_}'
            loop_config, teacher = load_config_and_model(dir_,
                                                         args.is_dd,
                                                         args.do_aug)
            loop_config.learner_c.arch = learner_arch
            # eval
            nm = f"{dir_}_{learner_arch}"
            res[nm] = []
            for i in tqdm(range(5)):
                _, acc, _, tm = eval_step(test_loader, loop_config, 
                                          teacher, 'cuda:0', args.n_epochs)
                res[nm].append({'acc': acc, 'time': tm})
                with open(args.res_pth, 'w') as f:
                    f.write(json.dumps(res))
                torch.cuda.empty_cache()             
        print(res)
