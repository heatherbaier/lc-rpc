from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.rpc import RRef, rpc_sync, rpc_async, remote
from torch.distributed.optim import DistributedOptimizer
import torch.distributed as dist
from torch import optim, nn
from threading import Lock

from environment import *
from dataloader import *

# from param_server import *
# from trainer_net import *


# The global parameter server instance.
param_server = None
# A lock to ensure we only have one parameter server.
global_lock = Lock()


# --------- Helper Methods --------------------

# On the local node, call a method with first arg as the value held by the
# RRef. Other args are passed in as arguments to the function called.
# Useful for calling instance methods. method could be any matching function, including
# class methods.
def _call_method(method, rref, *args, **kwargs):

    r"""
    a helper function to call a method on the given RRef
    """

    return method(rref.local_value(), *args, **kwargs)

# Given an RRef, return the result of calling the passed in method on the value
# held by the RRef. This call is done on the remote node that owns
# the RRef and passes along the given argument.
# Example: If the value held by the RRef is of type Foo, then
# remote_method(Foo.bar, rref, arg1, arg2) is equivalent to calling
# <foo_instance>.bar(arg1, arg2) on the remote node and getting the result
# back.

def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)


def load_trainer_data(train_dl):
    print("here in dl in rank: ", dist.get_rank())


# Main loop for trainers.
def run_worker(rank, train_dl):#, world_size, num_gpus, train_loader, test_loader):

    rpc.init_rpc(f"trainer_{dist.get_rank()}", rank = dist.get_rank(), world_size = int(os.environ['WORLD_SIZE']), rpc_backend_options = rpc.TensorPipeRpcBackendOptions(_transports = ["uv"], rpc_timeout = 5000))

    with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
        f.write(f"Worker rank {rank} initializing RPC IN RUN WORKER")   

    run_training_loop(dist.get_rank(), 0, train_dl, 0)


def get_parameter_server(num_gpus = 0):
    """
    Returns a singleton parameter server to all trainer processes
    """
    global param_server
    # Ensure that we get only one handle to the ParameterServer.
    with global_lock:
        if not param_server:
            # construct it once
            param_server = ParameterServer(num_gpus=num_gpus)
        return param_server


def run_training_loop(rank, num_gpus, train_loader, test_loader):

    # Runs the typical nueral network forward + backward + optimizer step, but
    # in a distributed fashion.
    net = TrainerNet(num_gpus = num_gpus)
    # Build DistributedOptimizer.
    param_rrefs = net.get_global_param_rrefs()
    opt = DistributedOptimizer(optim.SGD, param_rrefs, lr = 0.03)

    with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
        f.write("DONE UP TO HERE IN TRAINING LOOP!!!!\n")        

    print("DONE UP TO HERE IN TRAINING LOOP!!!!")

    envs = []
    for i in train_loader:
        muni_id, features, impath, lcpath, y = i
        env = Env(muni_id, features, y, impath, lcpath, id)
        envs.append(env)
        # ys.append(torch.tensor([y]))
        # features.append(env.get_features())
    # features, ys = torch.cat(features), torch.cat(ys).view(-1, 1)
    
    with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
        f.write("DONE LOADING IMAGERY ENVIRONMENTS IN RANK: " + str(dist.get_rank()) + ": " + str(len(envs)) + "\n")

    # train_dl = 

    # for 

    criterion = torch.nn.L1Loss()

    for i in range(len(envs)):

        with dist_autograd.context() as cid:

            try:

                model_output = net(envs[i].lc, envs[i].features, envs[i].im, envs[i].muni_id)

                loss = criterion(model_output, envs[i].y)

                dist_autograd.backward(cid, [loss])

                assert remote_method(
                    ParameterServer.get_dist_gradients,
                    net.param_server_rref,
                    cid) != {}

                opt.step(cid)

                with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
                    f.write("MODEL OUPUT IN RANK: " + str(dist.get_rank()) + ": " + str(model_output) + "\n")

            except:

                pass

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


class TrainerNet(nn.Module):

    def __init__(self, num_gpus=0):
        super().__init__()
        self.num_gpus = num_gpus
        self.param_server_rref = rpc.remote(
            "parameter_server", get_parameter_server, args=(num_gpus,))

    def get_global_param_rrefs(self):
        remote_params = remote_method(
            ParameterServer.get_param_rrefs,
            self.param_server_rref)
        return remote_params

    def forward(self, lc_im, census, im, muni_id):
        model_output = remote_method(
            ParameterServer.forward, self.param_server_rref, lc_im, census, im, muni_id)
        return model_output


import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
from torch import nn

from models import *

# --------- Parameter Server --------------------
class ParameterServer(nn.Module):
    def __init__(self, num_gpus = 0):
        super().__init__()
        model = lcModel()
        self.model = model
        self.input_device = "cpu"

    def forward(self, lc_im, census, im, muni_id):
        # inp = inp.to(self.input_device)
        out = self.model(lc_im, census, im, muni_id)
        # This output is forwarded over RPC, which as of 1.5.0 only accepts CPU tensors.
        # Tensors must be moved in and out of GPU memory due to this.
        # out = out.to("cpu")
        return out

    # Use dist autograd to retrieve gradients accumulated for this model.
    # Primarily used for verification.
    def get_dist_gradients(self, cid):
        grads = dist_autograd.get_gradients(cid)
        # This output is forwarded over RPC, which as of 1.5.0 only accepts CPU tensors.
        # Tensors must be moved in and out of GPU memory due to this.
        cpu_grads = {}
        for k, v in grads.items():
            k_cpu, v_cpu = k.to("cpu"), v.to("cpu")
            cpu_grads[k_cpu] = v_cpu
        return cpu_grads

    # Wrap local parameters in a RRef. Needed for building the
    # DistributedOptimizer which optimizes paramters remotely.
    def get_param_rrefs(self):
        param_rrefs = [rpc.RRef(param) for param in self.model.parameters()]
        return param_rrefs

