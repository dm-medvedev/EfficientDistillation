import sys
from torch.utils.data import IterableDataset, Dataset
import torch.nn as nn
from pathlib import Path
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import os.path as osp
import os
import json

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
from gm.utils import epoch, TensorDataset, distance_wb
from gtn.models.mnist_teacher import (Teacher, TeacherInputType,
                                      MNISTTeacher, MNISTTeacherConfig)
from my_experiments.teacher import DDTeacher
from badger_utils.torch.torch_utils import id_to_one_hot
from typing import Dict, Callable, Tuple
import torch.nn.functional as F
from pathlib import Path
from dataclasses import dataclass, asdict
import copy
from torch.optim.sgd import SGD


def prepare_data(teacher, step=None, class_nm=None):
    if step is not None:
        syn_target = teacher.input_target[step].view(-1)
        teacher_input = teacher.input_data[step]
    else:
        syn_target = teacher.input_target.view(-1)
        teacher_input = teacher.input_data.squeeze(0)
    if class_nm is not None:
        msk = syn_target == class_nm
        syn_target = syn_target[msk]
        teacher_input = teacher_input[msk]
    if isinstance(teacher, MNISTTeacher):
        ohe_target = id_to_one_hot(syn_target, teacher.config.target_classes)
        if teacher.training:
            teacher_output, _ = teacher(teacher_input, ohe_target)
        else:
            with torch.no_grad():
                teacher_output, _ = teacher(teacher_input, ohe_target)
    elif isinstance(teacher, DDTeacher):
        teacher_output = teacher_input
        if not teacher.training: teacher_output = teacher_output.detach()
    return teacher_output, syn_target


class TeacherIterDataset(IterableDataset):
    def __init__(self, teacher, cached=False):
        self.teacher = teacher
        self.cached = cached
        if self.cached:
            self.data, self.target = prepare_data(self.teacher)

    def __iter__(self):
        if self.cached:
            data, target = self.data, self.target
        else:
            data, target = prepare_data(self.teacher)
        yield data.cpu(), target.cpu()


class TeacherMapDataset(Dataset):
    def __init__(self, teacher):
        self.teacher = teacher
    
    def __getitem__(self, index):
        data, target = prepare_data(self.teacher, step=index)
        return data.cpu(), target.cpu()

    def __len__(self):
        return teacher.config.input_count


def evaluate_synset(net, images_train, labels_train, testloader, 
                    lr, batch_sz, param_augment, device, Epoch=1000):
    net = net.to(device)
    images_train = images_train.to(device)
    labels_train = labels_train.to(device)
    lr = float(lr)
    lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(device)

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=batch_sz, shuffle=True, num_workers=0)

    start = time.time()
    for ep in range(Epoch+1):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer, criterion, param_augment, device)
        if ep in lr_schedule:
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    time_train = time.time() - start
    loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, param_augment, device)

    return net, acc_train, acc_test, loss_train


def evaluate_net(net, trainloader, testloader, lr, 
                 param_augment, device, Epoch=1000, schedule=True,
                 epoch_to_eval=None):
    results = []
    net = net.to(device)
    lr = float(lr)
    lr_schedule = [Epoch//2+1] if schedule else []
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9,
                                weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().to(device)
    start = time.time()
    for ep in range(Epoch+1):
        loss_train, acc_train = epoch('train', trainloader, net, optimizer,
                                      criterion, param_augment, device)
        if ep in lr_schedule: # как плохо
            lr *= 0.1
            optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)
        if ep in epoch_to_eval:
          loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, param_augment, device)
          results.append((acc_train, acc_test, loss_train))
    time_train = time.time() - start
    loss_test, acc_test = epoch('test', testloader, net, optimizer, criterion, param_augment, device)
    return net, (acc_train, acc_test, loss_train) if len(results) == 0 else results


def batches4classes(loader, batch_size, num_classes, real_steps=1):
    X, y = next(x for x in loader)
    batches = {i: torch.zeros(size=(0, *X.shape[1:])) 
               for i in range(num_classes)}
    condition = lambda batches: any(len(batches[i]) < batch_size*real_steps 
                                    for i in batches)
    while condition(batches):
        for i in batches:
            batches[i] = torch.cat([batches[i], X[y == i]])[:batch_size*real_steps]
        X, y = next(x for x in loader)
    if real_steps > 1:
        for k in batches:
            batches[k] = [batches[k][i*batch_size:(i+1)*batch_size]
                          for i in range(real_steps)]
    return batches


def gm_loss(real_x, real_y, syn_x, syn_y, learner, normalize=False):
    # obtain gradient on original batch
    loss_real = F.cross_entropy(learner(real_x), real_y)
    orig_dy_dx = torch.autograd.grad(loss_real, learner.parameters())
    orig_dy_dx = list((_.detach().clone() for _ in orig_dy_dx))

    # obtain gradient on teacher's batch
    loss_syn = F.cross_entropy(learner(syn_x), syn_y)
    synth_dy_dx = torch.autograd.grad(loss_syn, learner.parameters(),
                                      create_graph=True)
    # gradient matching loss
    mloss = 0.0
    for g_pred, g_real in zip(synth_dy_dx, orig_dy_dx):
        mloss += distance_wb(g_pred, g_real)
    if normalize:
        mloss /= len(synth_dy_dx)
    return mloss, loss_real.item(), loss_syn.item()


def my_gm_loss(real_data, syn_x, syn_y, learner,
               normalize=False):
    # obtain gradient on original batch
    # learner.to('cpu')
    cp_learner = copy.deepcopy(learner)
    # cp_learner.to('cuda:0')
    cp_learner.train()
    local_optim = SGD(cp_learner.parameters(), lr=1.)
    for real_x, real_y in real_data:
        local_optim.zero_grad()
        loss_real = F.cross_entropy(cp_learner(real_x.to('cuda:0')), real_y.to('cuda:0'))
        loss_real.backward()
        local_optim.step()
    orig_dy_dx = []
    # cp_learner.to('cpu')
    for p1, p2 in zip(cp_learner.parameters(), learner.parameters()):
        orig_dy_dx.append((p2.data - p1.data).detach()) # .to('cuda:0')
    # del cp_learner
    # learner.to('cuda:0')
    # obtain gradient on teacher's batch
    loss_syn = F.cross_entropy(learner(syn_x), syn_y)
    synth_dy_dx = torch.autograd.grad(loss_syn, learner.parameters(),
                                      create_graph=True)
    # gradient matching loss
    mloss = 0.0
    for g_pred, g_real in zip(synth_dy_dx, orig_dy_dx):
        mloss += distance_wb(g_pred, g_real)
    if normalize:
        mloss /= len(synth_dy_dx)
    return mloss, loss_real.item(), loss_syn.item()


@dataclass
class LearnerConfig:
    # check gm/utils/get_network
    optim_cls: Callable
    optim_kwgs: Dict
    arch: str = "ConvNet"


@dataclass
class RealDataConfig:
    # check gm/utils/get_dataset
    channel: int = 1
    im_size: Tuple[int, int] = (28, 28)
    num_classes: int = 10


@dataclass
class AugConfig:
    # check gm/utils/get_daparam
    crop: int = 4
    scale: float = 0.2
    rotate: int = 45
    noise: float = 1e-3
    strategy: str = 'crop_scale_rotate'


@dataclass
class EvalConfig:
    # check gm/utils/get_daparam
    aug_c: Dict
    lr: float = 0.01
    epoch: int = 1_000


@dataclass
class LoopConfig:
    seed: int
    learner_c: LearnerConfig
    data_c: RealDataConfig
    eval_c: EvalConfig
    epochs: int = 1_000
    learned_curiculum: bool = False
    eval_it: int = 10

    @classmethod
    def load(cls, pth):
        with open(pth, 'r') as f:
            res = json.loads(f.read())
        res['learner_c'] = LearnerConfig(**res['learner_c'])
        res['data_c'] = RealDataConfig(**res['data_c'])
        if res['eval_c']['aug_c'] is not None:
            res['eval_c']['aug_c'] = asdict(AugConfig(**res['eval_c']['aug_c']))
        res['eval_c'] = EvalConfig(**res['eval_c'])
        return LoopConfig(**res)

    def dump(self, pth):
        with open(pth, 'w') as f:
            res = asdict(self)
            optim = self.learner_c.optim_cls
            optim = f'{optim.__module__}.{optim.__name__}'
            res['learner_c']['optim_cls'] = optim
            f.write(json.dumps(res)+'\n')


def get_writer(log_dir):
    assert not osp.exists(log_dir)
    return SummaryWriter(log_dir=log_dir, flush_secs=10)


def dump_results(teacher_config, teacher, 
                 teacher_dir: Path, config: LoopConfig, config_pth: Path):
    os.makedirs(teacher_dir, exist_ok=False)
    with torch.no_grad():
        teacher.eval()
        teacher.to('cpu')
        torch.save(teacher.state_dict(), teacher_dir/'model.bin')
    if isinstance(teacher_config, MNISTTeacherConfig):
        teacher_config.input_type = TeacherInputType.to_string(teacher_config.input_type)
    teacher_config = json.dumps(asdict(teacher_config))
    with open(teacher_dir/'config.json', 'w') as f: f.write(teacher_config+'\n')
    config.dump(config_pth)
