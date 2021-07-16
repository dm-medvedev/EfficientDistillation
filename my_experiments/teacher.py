import sys
sys.path.insert(0, '/home/dmitry/МАГА/Диплом/project/')

from gtn.models.mnist_teacher import Teacher
from typing import Optional, Tuple
from torch import Tensor
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class DDTeacherConfig:
    """
    Total number of images:
    num_classes * ipc * input_count
    """
    ipc: int = 10 # images per class
    input_count: int = 1
    channel: int = 1
    im_size: Tuple[int, int] = (28, 28)
    num_classes: int = 10


class DDTeacher(Teacher):
    def __init__(self, config:DDTeacherConfig, device:Optional[str]=None):
        super().__init__(device)
        self.config = config
        data = torch.rand((self.config.input_count, 
                           self.config.ipc * self.config.num_classes,
                           self.config.channel,
                           *self.config.im_size), device=self.device)
        self._input_data = nn.Parameter(data, True)
        input_target = torch.tensor([cls for _ in range(self.config.ipc) 
                                         for _ in range(self.config.input_count)
        			  				     for cls in range(self.config.num_classes)], 
                                    device=self.device)
        self.input_target = input_target.view(config.input_count, -1)


    @property
    def input_data(self) -> Tensor:
        return self._input_data
