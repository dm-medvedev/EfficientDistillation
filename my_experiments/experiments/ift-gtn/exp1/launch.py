import sys
sys.path.insert(0, '/home/dmitry/МАГА/Диплом/project/')

from torch.optim import SGD, Adam
import torch
from tqdm import tqdm
from pathlib import Path

from gtn.datasets.datasets import Datasets
from gtn.models.mnist_teacher import (MNISTTeacher, MNISTTeacherConfig,
                                      TeacherInputType)

from ift.optim import MetaOptimizer

from my_experiments.learning_loop import TeacherLearningLoop
from my_experiments.utils import (get_writer, dump_results, LearnerConfig, RealDataConfig,
                                  EvalConfig, LoopConfig)



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
    loop_c = LoopConfig(seed=42, # TODO: сделать так чтобы работал?
                        learner_c=learner_c,
                        data_c=data_c,
                        eval_c=eval_c,
                        epochs=100,
                        learned_curiculum=False,
                        eval_it=50)
    teacher_c = MNISTTeacherConfig(input_size=64,
                                   output_size=loop_c.data_c.im_size[0], # width of final image
                                   fc1_size=64, # 1024
                                   fc2_filters=32, # 128
                                   conv1_filters=16, # 64
                                   target_classes=loop_c.data_c.num_classes,
                                   input_count=1,
                                   batch_size=256,
                                   input_type=TeacherInputType.from_string('learned'))
    return loop_c, teacher_c

def get_teacher(device, teacher_c):
    teacher = MNISTTeacher(teacher_c, device=device)
    # teacher.learner_optim_params = None
    meta_optimizier = Adam(teacher.parameters(), lr=0.01)
    optimizer_teacher = MetaOptimizer(meta_optimizier, hpo_lr=0.01, 
                                      truncate_iter=3, max_grad_norm=4)
    return teacher, optimizer_teacher


def get_loader():
    train_loader = Datasets.mnist_dataloader(256, train=True)
    test_loader = Datasets.mnist_dataloader(512, train=False)
    return train_loader, test_loader


def main():
    loop_c, teacher_c = set_config()
    current_dir = Path(__file__).parent.absolute()/'try-1'
    writer = get_writer(current_dir/'log')
    device = torch.device('cuda:0')
    teacher, optimizer_teacher = get_teacher(device, teacher_c)
    train_loader, test_loader = get_loader()
    loop = TeacherLearningLoop(teacher, optimizer_teacher, loop_c,
                           train_loader, test_loader, writer, 
                           device=device)
    # cycle:
    for epoch in tqdm(range(1, loop_c.epochs + 1)):
        loop.ift_train(max_steps=10)
    # dump results
    dump_results(teacher_c, teacher, 
                 current_dir/'teacher',
                 loop_c, current_dir/'loop_c.json')


if __name__ == "__main__":
    main()
