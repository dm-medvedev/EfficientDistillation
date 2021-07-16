Наверное нужно задать всем на дистилляцию какое то определенное время, например 30 минут обучения (посчитав примерно сколько нужно будет итераций), а затем посмотреть результат

**higher-dd**: 
1. очень важно momentum побольше делать почему то

**higher-gtn**:
1. очень важно zero_grad было не забыть
1. У них обучение шло с config:
	> большой генератор, 
	> batch_size = 128,
	> 32 шага => 4,096 изображений
	> фиксированный curriculum
	> Качество: \~98%
1. У чуваков которые реализовывали (https://github.com/GoodAI/GTN):
	
	> *Experiment A*: batch_size: 32, inner_loop_steps: 20 
	(640 teacher-generated MNIST-images in total) 
	[обученный curriculum: acc \~97%: 5_000 эпох]
	[не обученный curriculum: acc \~92%: 5_000 эпох]
	
	> *Experiment B*: batch_size: 32, inner_loop_steps: 10 (320)
	[обученный curriculum: acc \~96%: 4_000 эпох]
	[обученный curriculum: acc \~94%: 1_000 эпох]
	[не обученный curriculum: acc \~90%: 4_000 эпох]
	[не обученный curriculum: acc \~90%: 1_000 эпох]
	
	> *Experiment C*: batch_size: 16, inner_loop_steps: 10 (160
	[обученный curriculum: acc \~95%: 5_000 эпох]
	[обученный curriculum: acc \~94%: 1_000 эпох]
	[не обученный curriculum: acc \~87%: 1_000 эпох]

**gm-dd**
1. Должно показать качество не хуже чем в статье (https://github.com/VICO-UoE/DatasetCondensation)
`python3 main.py  --dataset MNIST  --model ConvNet  --ipc 10 --data_path ../data --num_eval 1 --num_exp 1` ~ 16s/it
	|            | MNIST |
	 :-:         | :-:   |
	| 1 img/cls  | 91.7  | ~ 0.6 s / it
	| 10 img/cls | 97.4  | ~ 16 s/it
	| 50 img/cls | 98.8  | - очень много


**ift-dd**
1. ! Здесь нужно подробнее гиперпараметры изучить

**ift-gtn**
1. ! Здесь нужно подробнее гиперпараметры изучить

**gm-gtn**
1. ! Здесь нужно подробнее гиперпараметры изучить
