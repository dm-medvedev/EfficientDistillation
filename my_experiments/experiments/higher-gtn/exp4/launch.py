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

from ift.optim import MetaOptimizer

from my_experiments.learning_loop import TeacherLearningLoop
from my_experiments.utils import (get_writer, dump_results, LearnerConfig, RealDataConfig,
                                  EvalConfig, LoopConfig)
import copy



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
                        epoch=1)
    loop_c = LoopConfig(seed=42, # TODO: сделать так чтобы работал?
                        learner_c=learner_c,
                        data_c=data_c,
                        eval_c=eval_c,
                        epochs=1000,
                        learned_curiculum=False,
                        eval_it=20)
    teacher_c = MNISTTeacherConfig(input_size=64,
                                   output_size=loop_c.data_c.im_size[0], # width of final image
                                   fc1_size=64, #1024, 
                                   fc2_filters=32, # 128, 
                                   conv1_filters=16, # 64, 
                                   target_classes=loop_c.data_c.num_classes,
                                   input_count=1,
                                   batch_size=100,
                                   input_type=TeacherInputType.from_string('learned'))
    return loop_c, teacher_c


def get_teacher(device, teacher_c):
    teacher = MNISTTeacher(teacher_c, device=device)
    optimizer_teacher = Adam(teacher.parameters(), lr=0.01)
    return teacher, optimizer_teacher


def get_loader():
    dataset = Datasets.mnist_dataset(train=True)
    train_loader = DataLoader([dataset[i] for i in range(10_000, len(dataset))],
                              batch_size=256, shuffle=True, pin_memory=False,
                              num_workers=0)
    valid_loader = DataLoader([dataset[i] for i in range(10_000)],
                              batch_size=512, shuffle=False, pin_memory=False,
                              num_workers=12)
    # train_loader = Datasets.mnist_dataloader(256, train=True)
    # test_loader = Datasets.mnist_dataloader(512, train=False)
    return train_loader, valid_loader


def main():
    for i in range(2, 3): # думаю 3-х будет достаточно!
        loop_c, teacher_c = set_config()
        current_dir = Path(__file__).parent.absolute()/f'try-{i}'
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
        for epoch in tqdm(range(1, loop_c.epochs + 1)):
            acc = loop.higher_train(learner_steps=10)
            if (acc is not None) and (best_acc < acc):
                best_teacher = copy.deepcopy(loop.teacher).eval().to('cpu')
                # best_acc = acc

        # dump results
        dump_results(teacher_c, best_teacher, 
                     current_dir/'teacher',
                     loop_c, current_dir/'loop_c.json')


if __name__ == "__main__":
    main()
