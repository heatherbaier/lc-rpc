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
import numpy as np

from environment import *
from dataloader import *
from new_model import *
from utils import *




@record
def train(rank, world_size):

    if config.use_rpc:

        eval = Evaluator()
        eval_rref = eval.evaluator_rref

    envs, train_num = setup_data(world_size)

    # create model and move it to GPU with id rank
    model = lcNet()
    ddp_model = DDP(model, find_unused_parameters = config.find_unused_parameters)

    loss_fn = nn.L1Loss()
    optimizer = optim.SGD(ddp_model.parameters(), lr = config.lr)

    for epoch in range(0, config.num_epochs):

        for i in range(len(envs)):

            optimizer.zero_grad()

            coords = torch.FloatTensor(1, 2).uniform_(-1, 1)
            coords.requires_grad = True
            h_t = torch.zeros(1, config.core_hidden_size, dtype = torch.float, requires_grad = True)  
                
            last = False          
            for g in range(0, config.n_glimpses - 1):

                coords, h_t = ddp_model([envs[i].im, coords, h_t, envs[i].lc, envs[i].features, last])
                coords, h_t = coords.detach(), h_t.detach()

                # with open(config.log_name, "a") as f:
                #     f.write("RANK: " + str(dist.get_rank()) + " GLIMPSE " + str(g) + " COORDS: " + str(coords) + "\n")

            last = 'True'
            outputs = ddp_model([envs[i].im, coords, h_t, envs[i].lc, envs[i].features, last])

            # outputs = ddp_model(envs[i].im[:, :, 0:256, 0:256])
            print("OUTPUTS & Y IN RANK ", rank, ": ", outputs.item(), " ", envs[i].y.item())
            labels = envs[i].y
            loss = loss_fn(outputs, labels)
            loss.backward()

            for n,p in ddp_model.named_parameters():
                if p.grad is None:
                    print("None: ", n)

            optimizer.step()

            if config.use_rpc:
                coords = remote_method(Evaluator.collect_losses, eval_rref, loss, epoch)
            else:

                # Write the loss and prediction to a log file
                with open(config.train_records_name.format(epoch = str(epoch)), "a") as f:
                    f.write("LOSS: " + str(loss.item()) + " | PREDICTION: " + str(outputs.item()) + "\n")

                # If every rank has reported all of their losses, get the average and print it
                with open(config.train_records_name.format(epoch = str(epoch)), "r") as f:
                    rec = f.read()
                rec = [i.split(" | ") for i in rec.replace("\x00", "").splitlines() if i != ""]
                rec = [i for i in rec if len(i) == 2]
                rec = [float(i[0].strip("LOSS: ")) for i in rec]
                # rec = [float(i.split(" | ")[0].strip("LOSS: ")) for i in rec.replace("\x00", "").splitlines() if i != ""]
                if len(rec) == train_num:
                    loss_avg = np.mean(np.array(rec))
                    print("EPOCH: ", epoch, "  LOSS: ", loss_avg)
                    with open(config.log_name, "a") as f:
                        f.write("EPOCH: " + str(epoch) + "  LOSS: " + str(loss_avg))           




    time.sleep(20)

    cleanup()


if __name__ == "__main__":

    torch.autograd.set_detect_anomaly(True)

    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ["RANK"])

    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    dist.barrier()

    if config.use_rpc:
        rpc.init_rpc(f"worker_{dist.get_rank()}", rank = dist.get_rank(), world_size = int(os.environ['WORLD_SIZE']), rpc_backend_options = rpc.TensorPipeRpcBackendOptions(_transports=["uv"], rpc_timeout = 5000))
    else:
        if dist.get_rank() == 0:
            os.mkdir(config.records_dir)

    train(dist.get_rank(), int(os.environ['WORLD_SIZE']))
