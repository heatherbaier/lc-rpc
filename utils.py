from torch.distributed.elastic.multiprocessing.errors import record

import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torchvision import models
import time

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed.rpc as rpc
from threading import Lock

from environment import *
from dataloader import *
from new_model import *
from utils import *


# The evaluator instance.
evaluator = None

# A lock to ensure we only have one parameter server.
global_lock = Lock()


def setup(rank, world_size):
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def _call_method(method, rref, *args, **kwargs):

    r"""
    a helper function to call a method on the given RRef
    """

    return method(rref.local_value(), *args, **kwargs)


def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)


def setup_data(world_size):

    data = load_data(world_size)
    print(data)
    train_dl, val_dl = data.train_data, None
    train_dl = train_dl[dist.get_rank()]

    envs = []
    for i in train_dl:
        muni_id, features, impath, lcpath, y = i
        env = Env(muni_id, features, y, impath, lcpath, id)
        envs.append(env)

    return envs, data.train_num


def get_evaluator(placeholder):
    """
    Returns a singleton evaluator to all trainer processes
    """
    global evaluator
    # Ensure that we get only one handle to the ParameterServer.
    with global_lock:
        if not evaluator:
            # construct it once
            evaluator = Evaluator()
        return evaluator

# https://github.com/pytorch/examples/blob/master/imagenet/main.py
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Evaluator:

    def __init__(self):

        self.training_loss = 0
        self.validation_loss = 0
        self.num_train_collected = 0
        self.num_val_collected = 0
        self.evaluator_rref = rpc.remote(
            "worker_0", get_evaluator, args=(1,))
        self.epoch_tracker = {}
        self.num_train = 35

    def collect_losses(self, loss, epoch):

        # print("IN COLLECT LOSSES!")

        train = True 

        if epoch in self.epoch_tracker.keys():
            self.epoch_tracker[epoch].update(loss.item())
            if self.epoch_tracker[epoch].count == self.num_train:
                print("EPOCH: ", str(epoch), str(self.epoch_tracker[epoch].avg))
                with open(config.log_name, "a") as f:
                    f.write("EPOCH: " + str(epoch) + str(self.epoch_tracker[epoch].avg) + "\n")    
        else:
            self.epoch_tracker[epoch] = AverageMeter()
            self.epoch_tracker[epoch].update(loss.item())

        with open(config.log_name, "a") as f:
            f.write("IN COLLECT LOSSES WITH NUM = " + str(self.num_train_collected) + "\n")    

        return 0