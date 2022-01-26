#!/usr/bin/env python3
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.rpc import RRef, rpc_sync, rpc_async, remote
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.rpc as rpc
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from threading import Lock
import pprint
import torch
import sys
import io
import os

from config import get_config
# from param_server import *
# from trainer_net import *
from environment import *
from dataloader import *
from models import *
from utils2 import *

import warnings
warnings.filterwarnings("ignore")


config, _ = get_config()


def run_training_loop(rank, num_gpus, train_loader, test_loader):

    # Runs the typical nueral network forward + backward + optimizer step, but
    # in a distributed fashion.
    net = TrainerNet(num_gpus = num_gpus)
    # Build DistributedOptimizer.
    param_rrefs = net.get_global_param_rrefs()
    opt = DistributedOptimizer(optim.SGD, param_rrefs, lr=0.03)

    # for i, (data, target) in enumerate(train_loader):
    #     with dist_autograd.context() as cid:
    #         model_output = net(data)
    #         target = target.to(model_output.device)
    #         loss = F.nll_loss(model_output, target)
    #         if i % 5 == 0:
    #             print(f"Rank {rank} training batch {i} loss {loss.item()}")
    #         dist_autograd.backward(cid, [loss])
    #         # Ensure that dist autograd ran successfully and gradients were
    #         # returned.
    #         assert remote_method(
    #             ParameterServer.get_dist_gradients,
    #             net.param_server_rref,
    #             cid) != {}
    #         opt.step(cid)

    # print("Training complete!")
    # print("Getting accuracy....")
    # get_accuracy(test_loader, net)


@record
def setup_rpc():

    train_dl, val_dl = load_data(int(os.environ['WORLD_SIZE']))

    if dist.get_rank() == 0:
        rpc.init_rpc("parameter_server", rank = dist.get_rank(), world_size = int(os.environ['WORLD_SIZE']), rpc_backend_options = rpc.TensorPipeRpcBackendOptions(_transports=["uv"], rpc_timeout = 5000))
    elif dist.get_rank() == 1:
        rpc.init_rpc("evaluator", rank = dist.get_rank(), world_size = int(os.environ['WORLD_SIZE']), rpc_backend_options = rpc.TensorPipeRpcBackendOptions(_transports=["uv"], rpc_timeout = 5000))
        # evaluator = Evaluator()
    else:
        train_dl = train_dl[dist.get_rank()]
        print(f"trainer_{dist.get_rank()}")
        run_worker(dist.get_rank(), train_dl)

    rpc.shutdown()


    with open(config.log_name, "a") as f:
        f.write("DONE SETTING UP \n")  





if __name__ == '__main__':

    os.environ['TP_SOCKET_IFNAME'] = "eno1"
    os.environ['GLOO_SOCKET_IFNAME'] = "eno1"

    torch.autograd.set_detect_anomaly(True)

    env_dict = {
        k: os.environ[k]
        for k in (
            "LOCAL_RANK",
            "RANK",
            "GROUP_RANK",
            "WORLD_SIZE",
            "MASTER_ADDR",
            "MASTER_PORT",
            "TORCHELASTIC_RESTART_COUNT",
            "TORCHELASTIC_MAX_RESTARTS",
        )
    }

    with io.StringIO() as buff:
        print("======================================================", file=buff)
        print(
            f"Environment variables set by the agent on PID {os.getpid()}:", file=buff
        )
        pprint.pprint(env_dict, stream=buff)
        print("======================================================", file=buff)
        print(buff.getvalue())
        sys.stdout.flush()

    dist.init_process_group(backend="gloo")
    dist.barrier()

    print(
        (
            f"On PID {os.getpid()}, after init process group, "
            f"rank={dist.get_rank()}, world_size = {dist.get_world_size()}\n"
        )
    )

    setup_rpc()

