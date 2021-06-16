import math

import torch as T
from torch.optim.lr_scheduler import _LRScheduler


class GExponentialScheduler():
    def __init__(self, optimizer, patient_epoch=-1, inverse=False):
        self.optimizer = optimizer

        self.last_epoch = 0
        self.epoch_reduction = 0
        self.patient_epoch = patient_epoch
        self.inversed = inverse
        self.init_g = []

    def state_dict(self):
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ('optimizer', 'lr_lambda')
        }
        state_dict['lr_lambda'] = self.lr_lambda

        return state_dict

    def load_state_dict(self, state_dict):
        pass

    def step(self):
        if self.last_epoch == 0:
            for param_group in self.optimizer.param_groups:
                self.init_g.append(param_group['g'])

        self.last_epoch += 1

        # print("===== Step {} =====".format(self.last_epoch))
        if self.last_epoch > self.patient_epoch:
            self.epoch_reduction += 1
            for i, param_group in enumerate(self.optimizer.param_groups):
                if not self.inversed:
                    param_group['g'] = math.pow(self.init_g[i],
                                                self.epoch_reduction)
                else:
                    param_group['g'] = 1 - math.pow(1 - self.init_g[i],
                                                    self.epoch_reduction)

                # print('New G:', param_group['g'])
        else:
            param_group['g'] = 1.0
