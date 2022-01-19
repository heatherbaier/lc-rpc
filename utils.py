from torch.distributed.rpc import RRef, rpc_sync, rpc_async, remote
import torch.distributed.autograd as dist_autograd
from collections import namedtuple, deque
import torch.distributed.rpc as rpc
import torch.nn.functional as F
from torch import nn
import random
import torch




# # Use dist autograd to retrieve gradients accumulated for this model.
# # Primarily used for verification.
# def get_dist_gradients(self, cid):
#     grads = dist_autograd.get_gradients(cid)
#     # This output is forwarded over RPC, which as of 1.5.0 only accepts CPU tensors.
#     # Tensors must be moved in and out of GPU memory due to this.
#     cpu_grads = {}
#     for k, v in grads.items():
#         k_cpu, v_cpu = k.to("cpu"), v.to("cpu")
#         cpu_grads[k_cpu] = v_cpu
#     return cpu_grads

# # Wrap local parameters in a RRef. Needed for building the
# # DistributedOptimizer which optimizes paramters remotely.
# def get_param_rrefs(self):
#     param_rrefs = [rpc.RRef(param) for param in self.model.parameters()]
#     return param_rrefs



def calc_reward(true, pred):

    quants = [true * .10, 
            true * .20, 
            true * .30, 
            true * .40, 
            true * .50, 
            true * .60, 
            true * .70, 
            true * .80, 
            true * .90]
    
    if true == 0:
        quants = [true + 1, 
                true + 2, 
                true + 3,
                true + 4, 
                true + 5, 
                true + 6,
                true + 7, 
                true + 8, 
                true + 9]

    if (pred >= true - quants[0]) and (pred <= true + quants[0]):
        r = 100
    elif (pred >= true - quants[1]) and (pred <= true + quants[1]):
        r = 90
    elif (pred >= true - quants[2]) and (pred <= true + quants[2]):
        r = 80
    elif (pred >= true - quants[3]) and (pred <= true + quants[3]):
        r = 70        
    elif (pred >= true - quants[4]) and (pred <= true + quants[4]):
        r = 60
    elif (pred >= true - quants[5]) and (pred <= true + quants[5]):
        r = 50        
    elif (pred >= true - quants[6]) and (pred <= true + quants[6]):
        r = 40        
    elif (pred >= true - quants[7]) and (pred <= true + quants[7]):
        r = 30
    elif (pred >= true - quants[8]) and (pred <= true + quants[8]):
        r = 20      
    else:
        r = 0
        
    return r


Transition = namedtuple('Transition', ('census_feat', 'action', 'next_state', 'reward', 'lc_im'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def clean(epoch_preds, epoch_ys):
    
    epoch_preds = list(epoch_preds)
    epoch_preds.sort(key = lambda i: i[0])
    epoch_ys.sort(key = lambda i: i[0])

    epoch_preds = [i[1] for i in epoch_preds]
    epoch_ys = [i[1] for i in epoch_ys]

    preds = torch.tensor(epoch_preds).view(-1, 1)
    trues = torch.tensor(epoch_ys).view(-1, 1)

    return preds, trues


# def _call_method(method, rref, *args, **kwargs):

#     r"""
#     a helper function to call a method on the given RRef
#     """

#     return method(rref.local_value(), *args, **kwargs)


# def _remote_method(method, rref, *args, **kwargs):

#     r"""
#     a helper function to run method on the owner of rref and fetch back the
#     result using RPC
#     """

#     args = [method, rref] + list(args)
#     return rpc_sync(rref.owner(), _call_method, args = args, kwargs = kwargs)


# class Policy(nn.Module):

#     r"""
#     Borrowing the ``Policy`` class from the Reinforcement Learning example.
#     Copying the code to make these two examples independent.
#     See https://github.com/pytorch/examples/tree/master/reinforcement_learning
#     """

#     def __init__(self, h, w, outputs):

#         super(Policy, self).__init__()
#         self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
#         self.bn1 = nn.BatchNorm2d(16)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
#         self.bn2 = nn.BatchNorm2d(32)
#         self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
#         self.bn3 = nn.BatchNorm2d(32)

#         # Number of Linear input connections depends on output of conv2d layers
#         # and therefore the input image size, so compute it.
#         def conv2d_size_out(size, kernel_size = 5, stride = 2):
#             return (size - (kernel_size - 1) - 1) // stride  + 1
#         convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
#         convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
#         linear_input_size = convw * convh * 32
#         self.head = nn.Linear(linear_input_size, outputs)
#         self.mig_head = nn.Linear(linear_input_size, 1)
#         self.rnn = torch.nn.LSTM(input_size = linear_input_size, hidden_size = 256, num_layers = 1, batch_first = True)
#         self.fc = torch.nn.Linear(256, 1)


#     # Called with either one element to determine next action, or a batch
#     # during optimization. Returns tensor([[left0exp,right0exp]...]).
#     def forward(self, x, seq = None, select = False):
        
#         # x = x.to(device)
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = F.relu(self.bn3(self.conv3(x)))

#         if seq is not None:
#             seq = torch.cat( (seq, x.view(x.size(0), -1).unsqueeze(0)), dim = 1 )
#         else:
#             seq = x.view(x.size(0), -1).unsqueeze(0)

#         pred = self.rnn(seq)[0][:, -1, :].unsqueeze(0)

#         if select:
#             return self.head(x.view(x.size(0), -1)), self.fc(pred), x.view(x.size(0), -1).unsqueeze(0)
#         else:
#             return self.head(x.view(x.size(0), -1)), self.fc(pred)


