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

    def forward(self, x, census):
        # inp = inp.to(self.input_device)
        out = self.model(x, census)
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
