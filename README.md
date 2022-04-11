# Efficient Dataset Distillation using GTN, GM, IFT.

This project contains code of experiments for "Learning to Generate Synthetic Training Data using
Gradient Matching and Implicit Differentiation" paper, preprint text can be found [here](https://arxiv.org/abs/2203.08559). The paper was published in [AIST 2021](https://aistconf.org/) conference.

Using huge training datasets can be costly and inconvenient. This article explores various data distillation techniques that can reduce the amount of data required to successfully train deep networks. Inspired by recent ideas, we suggest new data distillation techniques based on generative teaching networks, gradient matching, and the Implicit Function Theorem. Experiments with the MNIST image classification problem show that the new methods are computationally more efficient than previous ones and allow to increase the performance of models trained on distilled data.


## Dataset Distillation
In machine learning, the purpose of [data distillation](https://arxiv.org/abs/1811.10959) is to compress the original dataset while maintaining the performance of the models trained on it. Generalizability is also needed: the ability of the dataset to train models of architectures that were not involved in the distillation process. Since training with less data is usually faster, distillation can be useful in practice. For example, it can be used to speed up a neural architecture search (NAS) task. Acceleration is achieved through the faster training of candidates.


## GTN
The idea first appeared in [paper](https://arxiv.org/abs/1912.07768), where authors suggested to use the generator as the teacher. The input of the generator is a concatenation of noise and one hot encoded label (for conditional generation). In the original paper, the authors use backpropagation through the student's learning process to train the generator, which is inconvenient for practical use due to high memory consumption, so in our paper, we show that the same or even better results can be achieved more effectively by using gradient matching or implicit differentiation.

## Gradient Matching
The gradient matching method was proposed in [paper](https://arxiv.org/pdf/2006.05929.pdf). The main difference with usual distillation is that we want not only to train the student to achieve a good performance on real data but also to get such a solution as if it was trained on real data. Our experiments with the MNIST benchmark show that selecting the correct size for the generator (GTN) allows to achieve better performance for gradient matching distillation. The original project can be found [here](https://github.com/VICO-UoE/DatasetCondensation).

## IFT
Implicit diffiretntiation method suggested in [paper](https://arxiv.org/abs/1911.02590) is based on implicit function theorem. Usage of IFT makes the dataset distillation procedure less computationally expensive, and we have found that using GTN with IFT improves generalizability for synthetic datasets generated with this method. The code we used as basis in our experiments can be found [here](https://github.com/AvivNavon/AuxiLearn).

## Citation

```
arXiv preprint arXiv:2203.08559
```
