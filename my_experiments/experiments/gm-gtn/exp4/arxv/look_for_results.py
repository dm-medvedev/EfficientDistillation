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
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
import time

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


def load_config_and_model(dir_, is_dd=True):
    # load config
    loop_config = LoopConfig.load(dir_/'loop_c.json')
    
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
    dump_nm = 'res.json'
    # with open(dump_nm, 'r') as f:
    #     res = json.loads(f.read())
    res = {}
    dataset = Datasets.mnist_dataset(train=True)
    test_loader = Datasets.mnist_dataloader(512, train=False)
    for dir_nm in glob.glob('./*'):
        if not os.path.isdir(dir_nm):
            continue
        for try_ in range(3):
            dir_ = Path(dir_nm)/f'try-{try_}'
            loop_config, teacher = load_config_and_model(dir_, is_dd=False)
            
            # eval
            res[str(dir_)] = []
            for i in tqdm(range(5)):
                epochs = 100 if dir_nm == 'ic_10' else 200
                _, acc, _, tm = eval_step(test_loader, loop_config, 
                                          teacher, 'cuda:0', epochs)
                res[str(dir_)].append({'acc': acc, 'time': tm})
                with open(dump_nm, 'w') as f:
                    f.write(json.dumps(res))
                
    print(res)
