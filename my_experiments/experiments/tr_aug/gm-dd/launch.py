import sys
sys.path.insert(0, '/home/dmitry/МАГА/Диплом/project/')

from torch.optim import SGD, Adam
import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

from gtn.datasets.datasets import Datasets
from my_experiments.teacher import DDTeacherConfig, DDTeacher

from ift.optim import MetaOptimizer

from my_experiments.learning_loop import TeacherLearningLoop
from my_experiments.utils import (get_writer, dump_results, LearnerConfig, RealDataConfig,
                                  EvalConfig, LoopConfig)
import copy
import time
import numpy as np
import os
from dataclasses import asdict
from my_experiments.utils import AugConfig


def set_config(ipc:int, epochs:int):
    learner_c = LearnerConfig(arch="ConvNet",
                              optim_cls=SGD,
                              optim_kwgs={"lr": 2e-2, 
                                          "momentum": 0.5})
    data_c = RealDataConfig(channel=1,
                            im_size=(28, 28),
                            num_classes=10,)
    eval_c = EvalConfig(aug_c=None,
                        lr=0.01,
                        epoch=100)
    loop_c = LoopConfig(seed=42, # TODO: сделать так чтобы работал?
                        learner_c=learner_c,
                        data_c=data_c,
                        eval_c=eval_c,
                        epochs=epochs, # 60
                        learned_curiculum=False,
                        eval_it=5)
    teacher_c = DDTeacherConfig(ipc=ipc, # images per class
                                input_count=1,
                                channel=1,
                                im_size=(28, 28),
                                num_classes=10)
    return loop_c, teacher_c


def get_teacher(device, teacher_c):
    teacher = DDTeacher(teacher_c, device=device)
    optimizer_teacher = SGD(teacher.parameters(), lr=0.1, momentum=0.5)
    return teacher, optimizer_teacher


def get_loader():
    dataset = Datasets.mnist_dataset(train=True)
    train_loader = DataLoader([dataset[i] for i in range(10_000, len(dataset))],
                              batch_size=256, shuffle=True, pin_memory=True,
                              num_workers=0)
    valid_loader = DataLoader([dataset[i] for i in range(10_000)],
                              batch_size=512, shuffle=False, pin_memory=True,
                              num_workers=12)
    # train_loader = Datasets.mnist_dataloader(256, train=True)
    # test_loader = Datasets.mnist_dataloader(512, train=False)
    return train_loader, valid_loader


def main(ipc:int, dir:Path, epoch:int, ous:int, ls:int):
    param_augment = asdict(AugConfig())
    print(param_augment)
    for i in range(3): # думаю 3-х будет достаточно!
        loop_c, teacher_c = set_config(ipc, epoch)
        current_dir = dir/f'try-{i}'
        writer = get_writer(current_dir/'log')
        device = torch.device('cuda:0')
        teacher, optimizer_teacher = get_teacher(device, teacher_c)
        train_loader, test_loader = get_loader()
        loop = TeacherLearningLoop(teacher, optimizer_teacher, loop_c,
                                   train_loader, test_loader, writer, 
                                   device=device)
        best_acc = -1
        best_teacher = None
        # cycle:
        time_stat = []
        for epoch in tqdm(range(1, loop_c.epochs + 1)):
            time_stat.append(time.time())
            acc = loop.gmpc_train(outer_steps=ous, 
                                  learner_steps=ls, 
                                  batch_size=256,
                                  normalize=False,
                                  param_augment=param_augment)
            assert param_augment is not None
            time_stat[-1] = time.time() - time_stat[-1]
            if (acc is not None) and (best_acc < acc):
                best_teacher = copy.deepcopy(loop.teacher).eval().to('cpu')
                best_acc = acc
        # dump results
        dump_results(teacher_c, best_teacher, 
                     current_dir/'teacher',
                     loop_c, current_dir/'loop_c.json')


if __name__ == "__main__":
    common_dir = Path(__file__).parent.absolute()
    usual_args = {'ipc':10, 'ous':10, 'ls':10, 
                  'epoch':110, 'dir': common_dir/f'out'}
    print(usual_args)
    os.makedirs(usual_args['dir'], exist_ok=True)
    main(**usual_args)
