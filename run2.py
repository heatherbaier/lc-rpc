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


def get_accuracy(test_loader, model):

    model.eval()
    correct_sum = 0
    # Use GPU to evaluate if possible
    device = torch.device("cuda:0" if model.num_gpus > 0
        and torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            out = model(data, -1)
            pred = out.argmax(dim=1, keepdim=True)
            pred, target = pred.to(device), target.to(device)
            correct = pred.eq(target.view_as(pred)).sum().item()
            correct_sum += correct

    print(f"Accuracy {correct_sum / len(test_loader.dataset)}")


@record
def setup_rpc():

    train_dl, val_dl = load_data(int(os.environ['WORLD_SIZE']))

    if dist.get_rank() == 0:

        rpc.init_rpc("parameter_server", rank = dist.get_rank(), world_size = int(os.environ['WORLD_SIZE']), rpc_backend_options = rpc.TensorPipeRpcBackendOptions(_transports=["uv"], rpc_timeout = 5000))

        # TestAgent(int(os.environ['WORLD_SIZE']))

        # run_parameter_server(0, int(os.environ['WORLD_SIZE']))

        # ob_rrefs = []
        # for ob_rank in range(1, int(os.environ['WORLD_SIZE'])):
        #     ob_info = rpc.get_worker_info("trainer_{}".format(ob_rank))
        #     print(ob_info)
        #     ob_rrefs.append(ob_info)

        # train_dl, val_dl = load_data(int(os.environ['WORLD_SIZE']))

        # print(train_dl)

    else:

        train_dl = train_dl[dist.get_rank()]

        print(f"trainer_{dist.get_rank()}")
        
        run_worker(dist.get_rank(), train_dl)

        # rpc.init_rpc(f"trainer_{dist.get_rank()}", rank = dist.get_rank(), world_size = int(os.environ['WORLD_SIZE']), rpc_backend_options = rpc.TensorPipeRpcBackendOptions(_transports = ["uv"], rpc_timeout = 5000))

    # with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
    #     f.write("OUTSIDE OF IF ELSE STUFF\n")     


    # if dist.get_rank() != 0:

    #     run_training_loop(dist.get_rank(), 0, 0, 0)
    
    # if 
    
    # dist.barrier()




        # init_worker(dist.get_rank(),
        #                 int(os.environ['WORLD_SIZE']), 
        #                 0,
        #                 None,
        #                 None)

    rpc.shutdown()


    with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
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


    if dist.get_rank() == 0:

        to_delete = ["/sciclone/home20/hmbaier/lc_v2/alog.txt", 
                     "/sciclone/home20/hmbaier/lc_v2/a_pred_log.txt", 
                     "/sciclone/home20/hmbaier/lc_v2/a_coords_log.txt", 
                     "/sciclone/home20/hmbaier/lc_v2/a_batch_log.txt"]

        for f in to_delete:
            if os.path.isfile(f):
                os.remove(f)


    setup_rpc()

