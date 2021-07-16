import sys
sys.path.insert(0, '/home/dmitry/МАГА/Диплом/project/')

from torch.optim import SGD, Adam
import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

from gtn.datasets.datasets import Datasets
from gtn.models.mnist_teacher import (MNISTTeacherConfig, MNISTTeacher,
                                      TeacherInputType)
from my_experiments.teacher import DDTeacherConfig, DDTeacher

from ift.optim import MetaOptimizer

from my_experiments.learning_loop import TeacherLearningLoop
from my_experiments.utils import (get_writer, dump_results, LearnerConfig, RealDataConfig,
                                  EvalConfig, LoopConfig)
import copy
import os
import numpy as np
import time

def set_config():
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
    
    ic, bsz, teacher_k, tp = 1, 100, 64, 'learned' # random, learned
    teacher_c = MNISTTeacherConfig(input_size=64,
                                   output_size=28, # width of final image
                                   fc1_size=teacher_k, # 1024
                                   fc2_filters=teacher_k//2, # 128
                                   conv1_filters=teacher_k//4, # 64
                                   target_classes=10,
                                   input_count=ic,
                                   batch_size=bsz,
                                   input_type=TeacherInputType.from_string(tp))

    # ic, ipc = 1, 10
    # teacher_c = DDTeacherConfig(ipc=ipc, # images per class
    #                             input_count=ic,
    #                             channel=1,
    #                             im_size=(28, 28),
    #                             num_classes=10)




    loop_c = LoopConfig(seed=42, # TODO: сделать так чтобы работал?
                        learner_c=learner_c,
                        data_c=data_c,
                        eval_c=eval_c,
                        epochs=1,
                        learned_curiculum=ic>1,
                        eval_it=5)
    return loop_c, teacher_c


def get_teacher(device, teacher_c):
    # teacher = DDTeacher(teacher_c, device=device)
    # optimizer_teacher = SGD(teacher.parameters(), lr=10)

    teacher = MNISTTeacher(teacher_c, device=device)
    optimizer_teacher = Adam(teacher.parameters(), lr=0.01)

    # teacher = DDTeacher(teacher_c, device=device)
    # meta_optimizier = SGD(teacher.parameters(), lr=10)
    # optimizer_teacher = MetaOptimizer(meta_optimizier, hpo_lr=0.01, 
    #                                   truncate_iter=10, max_grad_norm=4)

    # teacher = MNISTTeacher(teacher_c, device=device)
    # meta_optimizier = Adam(teacher.parameters(), lr=0.01)
    # optimizer_teacher = MetaOptimizer(meta_optimizier, hpo_lr=0.01, 
    #                                   truncate_iter=10, max_grad_norm=4)

    return teacher, optimizer_teacher


def get_loader():
    dataset = Datasets.mnist_dataset(train=True)
    train_loader = DataLoader([dataset[i] for i in range(10_000, len(dataset))],
                              batch_size=256, shuffle=True, pin_memory=True,
                              num_workers=0)
    valid_loader = DataLoader([dataset[i] for i in range(10_000)],
                              batch_size=512, shuffle=False, pin_memory=True,
                              num_workers=12)
    return train_loader, valid_loader


def main():
    loop_c, teacher_c = set_config()
    current_dir = Path(__file__).parent.absolute()
    writer = get_writer(current_dir/'to_delete')
    device = torch.device('cuda:0')
    teacher, optimizer_teacher = get_teacher(device, teacher_c)
    train_loader, test_loader = get_loader()

    loop = TeacherLearningLoop(teacher, optimizer_teacher, loop_c,
                               train_loader, test_loader, writer, 
                               device=device)

    ous, ls = 10, 10 # 10, 50
    loop.gmpc_train(outer_steps=ous,
                    learner_steps=ls,
                    batch_size=256)


    # ous, ls = 10, 10
    # loop.gmnpc_train(outer_steps=ous, learner_steps=ls, 
    #                  batch_size=256, normalize=True)

    # ls = 10
    # loop.higher_train(learner_steps=ls)

    # mx = 10
    # loop.ift_train(max_steps=mx)

    print(torch.cuda.max_memory_reserved()/1024/1024)



if __name__ == "__main__":
    main()
