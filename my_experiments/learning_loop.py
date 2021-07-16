import sys
sys.path.insert(0, '/home/dmitry/МАГА/Диплом/project/')

from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from gtn.models.mnist_teacher import Teacher, MNISTTeacherConfig
from typing import Optional
from my_experiments.utils import (TeacherIterDataset, prepare_data,
                                  batches4classes, my_gm_loss, gm_loss,
                                  evaluate_net, TeacherMapDataset)
from gm.utils import get_network, augment
import torch.nn.functional as F
from torch.nn.utils.clip_grad import clip_grad_norm_
from ift.optim import MetaOptimizer
import torch
from dataclasses import asdict
import higher
"""
наверное нужно будет просто доавить mode переменную
вместо отдельных классов для GTN и DD.
Или передавать необходимые функции: типа
prepare_gtn_data, GtnMapDataset, GtnIterDataset
"""

"""
Если у нас есть curiculum, то почему бы не сразу
использовать все обучаемые данные в одном батче?

Стохастичность помогает бороться с затратами 
по времени и по памяти когда мы обучаем learner, но как
она поможет в ift? когда нам нужно все данные обучаемые пропустить через него?
Согласен с тем что это помогает например в GM 
(для него нужно в первую очередь поэтому это организовать). Но в ift, думаю что
нет такого понятия как "обучаемый" шум в несколько батчей.
"""

class TeacherLearningLoop():
    def __init__(self, teacher:Teacher, optimizer_teacher, 
                 config, train_loader, test_loader, writer, 
                 device:Optional[str]='cpu'):
        self.teacher = teacher
        self.optimizer_teacher = optimizer_teacher
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.writer = writer
        self.device = device
        self.global_step = 0

    def get_learner(self):
        learner = get_network(self.config.learner_c.arch, 
                              **asdict(self.config.data_c))
        learner.to(self.device)
        learner.train()
        optim = self.config.learner_c.optim_cls
        optim = optim(learner.parameters(), **self.config.learner_c.optim_kwgs)
        return learner, optim

    def eval_step(self):
        learner = get_network(self.config.learner_c.arch, 
                              **asdict(self.config.data_c))
        self.teacher.eval()
        if self.config.learned_curiculum:
            trainloader = TeacherMapDataset(self.teacher)
        else:
            trainloader = TeacherIterDataset(self.teacher)
        param_augment = self.config.eval_c.aug_c
        res = evaluate_net(learner, trainloader, self.test_loader, 
                           self.config.eval_c.lr, param_augment, 
                           self.device, Epoch=self.config.eval_c.epoch)
        _, acc_train, acc_test, loss_train = res
        return acc_train, acc_test, loss_train

    def test_step(self, learner):
        """
        for debug
        """
        correct = 0
        count = 0
        for data, target in self.test_loader:
            data, target = data.to(self.device), target.to(self.device)
            output = learner(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            count += len(data)
        accuracy = correct / count
        return accuracy

    def gmpc_train(self, outer_steps, learner_steps, 
                   batch_size=256, normalize=True, 
                   param_augment=None):
        """
        gradient matching with per class loss
        """
        learner, optim = self.get_learner()
        for outer_step in range(outer_steps):
            self.teacher.train()
            real_batches = batches4classes(self.train_loader, batch_size,
                                           self.config.data_c.num_classes)
            log = {'learner real loss': 0.0,
                   'learner synthetic loss': 0.0}
            grad_diff = 0.0
            for class_nm in real_batches:
                # real data
                real_x = real_batches[class_nm]
                if param_augment is not None:
                    real_x = augment(real_x, param_augment, device='cpu')
                real_x = real_x.to(self.device)
                real_y = torch.ones(len(real_x), dtype=torch.int64,
                                    device = self.device)*class_nm
                # сюда можно будет добавить random choice
                step = (outer_step % self.teacher.config.input_count) \
                       if self.config.learned_curiculum else None
                syn_x, syn_y = prepare_data(self.teacher, step, class_nm)
                mloss, real_loss, syn_loss = gm_loss(real_x, real_y, syn_x, 
                                                     syn_y, learner, normalize)
                log['learner real loss'] += real_loss
                log['learner synthetic loss'] += syn_loss
                grad_diff += mloss
            if normalize:
                grad_diff /= len(real_batches)
            log['learner real loss'] /= len(real_batches)
            log['learner synthetic loss'] /= len(real_batches)
            log['gradient matching loss'] = grad_diff.item()

            # update teacher
            self.optimizer_teacher.zero_grad()
            grad_diff.backward()
            log['teacher grad norm'] = clip_grad_norm_(self.teacher.parameters(),
                                                       max_norm=10 if normalize else 1_000)
            self.optimizer_teacher.step()
            # update learner
            with torch.no_grad():
                self.teacher.eval()
                step = (outer_step % self.teacher.config.input_count) \
                       if self.config.learned_curiculum else None
                syn_x, syn_y = prepare_data(self.teacher, step)
            for _ in range(learner_steps):
                optim.zero_grad()
                loss = F.cross_entropy(learner(syn_x), syn_y)
                loss.backward()
                optim.step()
            with torch.no_grad():
                learner.eval()
                res = learner(syn_x).argmax(1) == syn_y
                log["learner train accuracy"] = res.cpu().numpy().mean()
        # logging
        for key in log:
            self.writer.add_scalar(f'train/{key}', log[key], 
                                   self.global_step)
        with torch.no_grad():
            self.teacher.eval()
            if self.global_step % self.config.eval_it == 0:
                self.writer.add_scalar('debug/test accuracy', global_step=self.global_step,
                                       scalar_value=self.test_step(learner))
        self.global_step += 1
        if self.global_step % self.config.eval_it == 0:
            acc_train, acc_test, loss_train = self.eval_step()
            self.writer.add_scalar('test/test accuracy', global_step=self.global_step,
                                   scalar_value=acc_test)
            return acc_test

    def higher_train(self, learner_steps, param_augment=None):
        learner, optim = self.get_learner()
        self.teacher.train()
        self.optimizer_teacher.zero_grad()
        log = {}
        with higher.innerloop_ctx(learner, optim) as (flearner, diffopt):
            # inner loop
            for in_step in range(learner_steps):
                step = (in_step % self.teacher.config.input_count) \
                       if self.config.learned_curiculum else None
                syn_x, syn_y = prepare_data(self.teacher, step)
                loss = F.cross_entropy(flearner(syn_x), syn_y)
                diffopt.step(loss)

            log['inner loss'] = loss.item()

            # obtain meta loss
            real_x, real_y = next(x for x in self.train_loader) # TODO: может быть баг
            if param_augment is not None:
                real_x = augment(real_x, param_augment, device='cpu')
            real_x, real_y = real_x.to(self.device), real_y.to(self.device)
            loss = F.cross_entropy(flearner(real_x), real_y)
            loss.backward()
            log['meta loss'] = loss.item()
            # breakpoint()
            with torch.no_grad():
                flearner.eval()
                res = flearner(syn_x).argmax(1) == syn_y
                log["learner train accuracy"] = res.cpu().numpy().mean()

        # norm = torch.norm(self.teacher._input_data.grad.view(10, -1), dim=1)
        # d = {f'layer {i}': g for i, g in enumerate(norm)}
        # self.writer.add_scalars(f'train/grad', d, 
        #                         self.global_step)
        log[f'teacher grad norm'] = clip_grad_norm_(self.teacher.parameters(), max_norm=10)
        self.optimizer_teacher.step()
        for key in log:
            self.writer.add_scalar(f'train/{key}', log[key], 
                                   self.global_step)
        if self.global_step % self.config.eval_it == 0:
            self.writer.add_scalar('debug/test accuracy', global_step=self.global_step,
                                   scalar_value=self.test_step(flearner))
        self.global_step += 1
        if self.global_step % self.config.eval_it == 0:
            acc_train, acc_test, loss_train = self.eval_step()
            self.writer.add_scalar('test/test accuracy', global_step=self.global_step,
                                   scalar_value=acc_test)
            return acc_test

    def ift_train(self, max_steps=10, epsilon=1e-4, tr_est_steps=1, param_augment=None):
        """
        нужно реализовать поддержку обучаемого шума?
        хотя кажется для одного шага она и так присутствует;
        спрашивается, а нужно ли больше?

        TODO: tr_est_steps - зависит от того обучаемый ли шум!
              поэтому нужно его как то заменить!

        TODO: обучение можно сделать более быстрым, 
        если начальная инициализация w будет равняться прошлой w*?
        """
        assert isinstance(self.optimizer_teacher, MetaOptimizer)
        learner, optim = self.get_learner()
        self.teacher.eval()
        log = {}
        # inner loop: obtain w^*
        norm0 = None
        for step in range(max_steps):
            # step = step if self.config.learned_curiculum else None
            # здесь нужно подумать как конвертировать получше
            syn_x, syn_y = prepare_data(self.teacher)
            loss = F.cross_entropy(learner(syn_x), syn_y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            norm = sum(p.grad.data.norm(2)**2 for p in learner.parameters())
            if norm0 is None: norm0 = norm
            if norm < (norm0 * epsilon): break
        log['Lw* grad norm'] = norm
        with torch.no_grad():
            learner.eval()
            res = learner(syn_x).argmax(1) == syn_y
            log["learner train accuracy"] = res.cpu().numpy().mean()
        learner.train()

        # здесь нужно взависимости от config.learned_curiculum, оценивать
        # или нужно подумать о другом неком параметре
        # estimate train loss
        self.teacher.train()
        train_loss = 0.
        for step in range(tr_est_steps):
            """
            здесь наблюдается рост в требуемой памяти
            поэтому этот цикл вряд ли нужен
            """ 
            syn_x, syn_y = prepare_data(self.teacher)
            train_loss += F.cross_entropy(learner(syn_x), syn_y)
        train_loss /= tr_est_steps
        log['train_loss'] = train_loss.item()

        # estimate validation loss: заметим, что тут я не делаю цикла!
        real_x, real_y = next(x for x in self.train_loader)
        if param_augment is not None:
            real_x = augment(real_x, param_augment, device='cpu')
        real_x, real_y = real_x.to(self.device), real_y.to(self.device)
        validation_loss = F.cross_entropy(learner(real_x), real_y)
        log['validation_loss'] = validation_loss.item()

        # step
        self.optimizer_teacher.step(train_loss, validation_loss, list(learner.parameters()),
                                    list(self.teacher.parameters()), log=log)
        for key in log:
            self.writer.add_scalar(f'train/{key}', log[key], 
                                   self.global_step)
        self.global_step+=1
        if self.global_step % self.config.eval_it == 0:
            acc_train, acc_test, loss_train = self.eval_step()
            self.writer.add_scalar('test/test accuracy', global_step=self.global_step,
                                   scalar_value=acc_test)
            return acc_test


    def my_gmpc_train(self, outer_steps=10, learner_steps=10,
                      real_steps=10, batch_size=256, normalize=True):
        """
        gradient matching with per class loss
        """
        learner, optim = self.get_learner()
        for outer_step in range(outer_steps):
            self.teacher.train()
            real_batches = batches4classes(self.train_loader, batch_size,
                                           self.config.data_c.num_classes,
                                           real_steps=real_steps)
            log = {'learner real loss': 0.0,
                   'learner synthetic loss': 0.0}
            grad_diff = 0.0
            for class_nm in real_batches:
                # real data
                real_data = [(real_x, torch.ones(len(real_x), dtype=torch.int64)*class_nm)
                             for real_x in real_batches[class_nm]]
                # сюда можно будет добавить random choice
                step = (outer_step % self.teacher.config.input_count) \
                       if self.config.learned_curiculum else None
                syn_x, syn_y = prepare_data(self.teacher, step, class_nm)
                mloss, real_loss, syn_loss = my_gm_loss(real_data, syn_x, syn_y, 
                                                        learner, normalize)
                log['learner real loss'] += real_loss
                log['learner synthetic loss'] += syn_loss
                grad_diff += mloss
            if normalize:
                grad_diff /= len(real_batches)
            log['learner real loss'] /= len(real_batches)
            log['learner synthetic loss'] /= len(real_batches)
            log['gradient matching loss'] = grad_diff.item()

            # update teacher
            self.optimizer_teacher.zero_grad()
            grad_diff.backward()
            log['teacher grad norm'] = clip_grad_norm_(self.teacher.parameters(),
                                                       max_norm=10 if normalize else 1_000)
            self.optimizer_teacher.step()
            # update learner
            with torch.no_grad():
                self.teacher.eval()
                step = (outer_step % self.teacher.config.input_count) \
                       if self.config.learned_curiculum else None
                syn_x, syn_y = prepare_data(self.teacher, step)
            for _ in range(learner_steps):
                optim.zero_grad()
                loss = F.cross_entropy(learner(syn_x), syn_y)
                loss.backward()
                optim.step()
            with torch.no_grad():
                learner.eval()
                res = learner(syn_x).argmax(1) == syn_y
                log["learner train accuracy"] = res.cpu().numpy().mean()
        # logging
        for key in log:
            self.writer.add_scalar(f'train/{key}', log[key], 
                                   self.global_step)
        with torch.no_grad():
            self.teacher.eval()
            if self.global_step % self.config.eval_it == 0:
                self.writer.add_scalar('debug/test accuracy', global_step=self.global_step,
                                       scalar_value=self.test_step(learner))
        del learner
        del optim
        self.global_step += 1
        if self.global_step % self.config.eval_it == 0:
            acc_train, acc_test, loss_train = self.eval_step()
            self.writer.add_scalar('test/test accuracy', global_step=self.global_step,
                                   scalar_value=acc_test)
            return acc_test

    def gmnpc_train(self, outer_steps, learner_steps, 
                    batch_size=256, normalize=True):
        """
        gradient matching with per class loss
        """
        learner, optim = self.get_learner()
        for outer_step in range(outer_steps):
            self.teacher.train()
            log = {'learner real loss': 0.0,
                   'learner synthetic loss': 0.0}
            # real data
            real_x, real_y = next(x for x in self.train_loader)
            real_x, real_y = real_x.to(self.device), real_y.to(self.device)
            # сюда можно будет добавить random choice
            step = (outer_step % self.teacher.config.input_count) \
                   if self.config.learned_curiculum else None
            syn_x, syn_y = prepare_data(self.teacher, step)
            mloss, real_loss, syn_loss = gm_loss(real_x, real_y, syn_x, 
                                                 syn_y, learner, normalize)
            log['learner real loss'] += real_loss
            log['learner synthetic loss'] += syn_loss
            grad_diff = mloss
            log['gradient matching loss'] = grad_diff.item()

            # update teacher
            self.optimizer_teacher.zero_grad()
            grad_diff.backward()
            log['teacher grad norm'] = clip_grad_norm_(self.teacher.parameters(),
                                                       max_norm=10 if normalize else 1_000)
            self.optimizer_teacher.step()
            # update learner
            with torch.no_grad():
                self.teacher.eval()
                step = (outer_step % self.teacher.config.input_count) \
                       if self.config.learned_curiculum else None
                syn_x, syn_y = prepare_data(self.teacher, step)
            for _ in range(learner_steps):
                optim.zero_grad()
                loss = F.cross_entropy(learner(syn_x), syn_y)
                loss.backward()
                optim.step()
            with torch.no_grad():
                learner.eval()
                res = learner(syn_x).argmax(1) == syn_y
                log["learner train accuracy"] = res.cpu().numpy().mean()
        # logging
        for key in log:
            self.writer.add_scalar(f'train/{key}', log[key], 
                                   self.global_step)
        with torch.no_grad():
            self.teacher.eval()
            if self.global_step % self.config.eval_it == 0:
                self.writer.add_scalar('debug/test accuracy', global_step=self.global_step,
                                       scalar_value=self.test_step(learner))
        self.global_step += 1
        if self.global_step % self.config.eval_it == 0:
            acc_train, acc_test, loss_train = self.eval_step()
            self.writer.add_scalar('test/test accuracy', global_step=self.global_step,
                                   scalar_value=acc_test)
            return acc_test
