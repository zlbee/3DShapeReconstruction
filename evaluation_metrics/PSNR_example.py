from collections import OrderedDict

import torch
from torch import nn, optim

from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *
import numpy as np

# create default evaluator for doctests

def eval_step(engine, batch):
    return batch

default_evaluator = Engine(eval_step)

# create default optimizer for doctests

param_tensor = torch.zeros([1], requires_grad=True)
default_optimizer = torch.optim.SGD([param_tensor], lr=0.1)

# create default trainer for doctests
# as handlers could be attached to the trainer,
# each test must define his own trainer using `.. testsetup:`

def get_default_trainer():

    def train_step(engine, batch):
        return batch

    return Engine(train_step)

# create default model for doctests

default_model = nn.Sequential(OrderedDict([
    ('base', nn.Linear(4, 2)),
    ('fc', nn.Linear(2, 1))
]))

manual_seed(666)

psnr = PSNR(data_range=1.0)
psnr.attach(default_evaluator, 'psnr')

l = []
for i in range(5):
    pred = np.load('./test_data/control_group/predicted_shape%02d.npz' % i)
    true = np.load('./test_data/control_group/real_shape%02d.npz' % i).astype(np.float32)
    img1 = torch.from_numpy(pred)
    img2 = torch.from_numpy(true)

    state = default_evaluator.run([[img1, img2]])
    result = state.metrics['psnr']
    l.append(result)

    print(result)
print(sum(l)/len(l))