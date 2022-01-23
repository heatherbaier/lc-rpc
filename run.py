#!/usr/bin/env python3
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.rpc import RRef, rpc_sync, rpc_async, remote
from torch.distributions import Categorical
import torch.autograd.profiler as profiler
import torch.distributed.rpc as rpc
import torch.distributed as dist
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
import torch.nn as nn
import numpy as np
import argparse
import pprint
import torch
import sys
import io
import os

from config import get_config
from environment import *
from dataloader import *
# from earth_env import *
from models import *
from utils import *

import warnings
warnings.filterwarnings("ignore")


config, _ = get_config()

TOTAL_EPISODE_STEP = 5000
AGENT_NAME = "agent"
OBSERVER_NAME = "observer{}"

parser = argparse.ArgumentParser(description='PyTorch RPC RL example')
parser.add_argument('--world-size', type = int, default = 4, metavar = 'W',
                    help='world size for RPC, rank 0 is the agent, others are observers')
parser.add_argument('--gamma', type = float, default = 0.99, metavar = 'G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type = int, default = 543, metavar = 'N',
                    help='random seed (default: 543)')
parser.add_argument('--log-interval', type = int, default = 10, metavar = 'N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

torch.manual_seed(args.seed)


def _call_method(method, rref, *args, **kwargs):

    r"""
    a helper function to call a method on the given RRef
    """

    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method, rref, *args, **kwargs):

    r"""
    a helper function to run method on the owner of rref and fetch back the
    result using RPC
    """

    args = [method, rref] + list(args)
    return rpc_sync(rref.owner(), _call_method, args = args, kwargs = kwargs)


def _remote_method_async(method, rref, *args, **kwargs):

    r"""
    a helper function to run method on the owner of rref and fetch back the
    result using RPC
    """

    args = [method, rref] + list(args)
    return remote(rref.owner(), _call_method, args = args, kwargs = kwargs)




class Observer:

    r"""
    An observer has exclusive access to its own environment. Each observer
    captures the state from its environment, and send the state to the agent to
    select an action. Then, the observer applies the action to its environment
    and reports the reward to the agent.
    It is true that CartPole-v1 is a relatively inexpensive environment, and it
    might be an overkill to use RPC to connect observers and trainers in this
    specific use case. However, the main goal of this tutorial to how to build
    an application using the RPC API. Developers can extend the similar idea to
    other applications with much more expensive environments.
    """

    def __init__(self):

        self.id = rpc.get_worker_info().id
        self.train_envs, self.val_envs = [], []
        # self.target_net = Policy(indim = 117, outdim = 23)
        # self.conv_net = ConvNet(resnet = models.resnet18(), n_glimpses = config.n_glimpses)
        self.lc_keys = self.read_lc_map()
        self.n_glimpses = config.n_glimpses
        self.lc_ids = self.read_lc_map()
        # self.criterion = torch.nn.L1Loss()
        # self.optimizer = torch.optim.Adam(self.conv_net.parameters(), lr = 0.01)


        self.lc_model = lcModel()
        self.criterion = torch.nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.lc_model.parameters(), lr = 0.01)


        with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
            f.write("Observer with ID: " + str(self.id) + "\n" + "WORKER INFO: " + str(rpc.get_worker_info()))               

    def read_lc_map(self):

        """
        Read in the land cover mapping JSON
        """

        with open(config.lc_map, "r") as f:
            self.lc_map = json.load(f)
        return torch.tensor([int(i) for i in list(self.lc_map.keys())])

    def load_data(self, data, validation):

        """
        Here we recieve an asynchrous call from the Agent with a list of training data batches. Select just 
        the batch that matches the current observers ID value (minus 1) and load it as the observer's data.
        Then, we load the environment's for each of the images in our dataset.
        """

        self.data = data[self.id - 1]

        with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
            f.write("NUMBER OF IMAGES IN OBSERVER " + str(self.id) + ": " + str(len(self.data)) + "\n")

        self.envs, self.features, self.ys = [], [], []
        for i in self.data:
            muni_id, features, impath, lcpath, y = i
            env = Env(muni_id, features, y, impath, lcpath, self.id)
            self.envs.append(env)
            self.ys.append(torch.tensor([y]))
            self.features.append(env.get_features())
        self.features, self.ys = torch.cat(self.features), torch.cat(self.ys).view(-1, 1)


    def step(self, agent_rref):

        """
        First choose the LC cover classes to sample glimpses from
        Next, refer to the environemnts to sample glimpses form the selected LC's
        Finaly, send those glimpses to the agent to process within the model
        """

        # Grab the land cover choices from the model
        # lc_output = self.target_net(self.features)
        # lc_output = _remote_method(Agent.select_lc, agent_rref, self.features)
        # (_, lc_choices) = torch.topk(lc_output, self.n_glimpses)

        # Send the land cover choices to the environments and retrieve the glimpse data
        glimpses = []
        for i in range(len(self.envs)):

            try:

                pred, edited = self.lc_model(self.envs[i].lc, self.envs[i].features, self.envs[i].im, self.envs[i].muni_id)
                loss = self.criterion(pred, self.envs[i].y)

                with open("/sciclone/home20/hmbaier/lc_v2/a_pred_log.txt", "a") as f:
                    f.write("PREDICTION: " + str(pred.item()) + "  Y: " + str(self.envs[i].y.item()) + "  LOSS: " + str(loss.item()) + "  MUNI ID: " + str(self.envs[i].muni_id) + " " + str(edited) + "\n")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # self.training_losses.append(loss.item())

            except Exception as e:

                with open("/sciclone/home20/hmbaier/lc_v2/a_batch_log.txt", "a") as f:
                    f.write("BAD MUNI ID: " + str(self.envs[i].muni_id) + " " + str(edited) + "\n" + str(e) + "\n")




            # _remote_method_async(Agent.select_lc, agent_rref, self.envs[i].lc, self.envs[i].features, self.envs[i].im, self.envs[i].y, self.envs[i].muni_id)

        #     coords = _remote_method(Agent.select_lc, agent_rref, self.envs[i].lc, self.envs[i].features)

        #     # with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
        #     #     f.write("NORM COORDS: " + str(coords) + "\n")

        #     glimpse = self.envs[i].get_glimpses(coords.detach().clone())

        # #     glimpses.append(glimpse)
        # # glimpses = torch.cat(glimpses)

        #     _remote_method_async(Agent.optimize_conv, agent_rref, glimpse, self.envs[i].y, self.envs[i].features, coords, self.envs[i].lc)

        # # Send the glimpses through the conv net
        # output = self.conv_net(glimpses)

        # loss = self.criterion(output, self.ys)
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        # with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
        #     f.write("PREDICTION: " + str(self.id) + ": " + str(output) + "\n")

        # with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
        #     f.write("LOSS: " + str(self.id) + ": " + str(loss) + "\n")            



class Agent:

    def __init__(self, world_size, config):

        self.ob_rrefs = []
        self.agent_rref = RRef(self)
        self.rewards = {}
        self.saved_log_probs = {}

        self.lc_model = lcModel()
        self.criterion = torch.nn.L1Loss()
        self.optimizer = torch.optim.Adam(self.lc_model.parameters(), lr = 0.01)


        # # Land Cover model setup
        # # self.lc_net_notupdated = Policy(indim = 117, outdim = 23)
        # # self.lc_net_updated = Policy(indim = 117, outdim = 23)
        # self.lc_net_notupdated = lcNet(models.resnet18())
        # # self.lc_net_notupdated.fc = torch.nn.Linear(512, 2)
        # self.lc_net_updated = lcNet(models.resnet18())
        # # self.lc_net_updated.fc = torch.nn.Linear(512, 2)        
        # self.lc_optimizer = torch.optim.Adam(self.lc_net_updated.parameters(), lr = 0.001)
        # self.lc_criterion = torch.nn.L1Loss()


        """
        conv_net_notupdated: Only copy the trained paremeters to this every 10 epochs
        conv_net_updated: This is the one that gets optimized
        """
        # self.conv_net_notupdated = ConvNet(resnet = models.resnet18(), n_glimpses = config.n_glimpses)
        # self.conv_net_updated = ConvNet(resnet = models.resnet18(), n_glimpses = config.n_glimpses)
        # self.conv_optimizer = torch.optim.Adam(self.conv_net_updated.parameters(), lr = 0.001)
        # self.conv_criterion = torch.nn.L1Loss()



        self.memory = []







        # self.n_actions = 5
        # self.policy = Policy(128, 128, self.n_actions)
        # self.target_net = Policy(128, 128, self.n_actions)
        # self.optimizer = optim.Adam(self.policy.parameters(), lr = 1e-2)
        # self.criterion = nn.L1Loss()
        # self.eps = np.finfo(np.float32).eps.item()
        # self.running_reward = 0
        # # self.reward_threshold = gym.make('CartPole-v1').spec.reward_threshold
        # self.config = config
        # self.steps_done = 0
        # self.eps_threshold = .9
        # self.memory = []
        # self.limit = 1000
        # self.to_tens = transforms.ToTensor()
        # self.preds_record, self.done_tracker = {}, {}
        # self.world_size = world_size

        # Load all of the data into the Agent on Rank 0
        self.train_dl, self.val_dl = self.load_data(world_size)

        # For each of the observers in the remote world, set up their information in the agent
        for ob_rank in range(1, world_size):
            ob_info = rpc.get_worker_info(OBSERVER_NAME.format(ob_rank))
            self.ob_rrefs.append(remote(ob_info, Observer))
            self.rewards[ob_info.id] = []
            self.saved_log_probs[ob_info.id] = []

        # Make an RPC call to distribute a data batch to each of the Observers
        self.distribute_data(self.train_dl, validation = False)
        # self.distribute_data(self.val_dl, validation = True)


    def distribute_data(self, data, validation = False):

        """
        TO-DO: FOR NOW THIS IS A BLOCKING SYNC CALL SINCE WE NEED THE IMAGE TO BE LOADED IN ORDER TO 
        ACTUALLY CALL RUN_EPISODE LATER ON. TRY TO FIND A WAY TO DO THE LIKE PROCESS.JOIN THING SO WE
        CAN GET THE LOADING STARTED ON ALL OBSERVERS SIMOUTANESOULY AND THEN WAIT FOR THEM ALL TO END
        """

        with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
            f.write("In distribute data function!" + "\n")

        futs = []

        # with profiler.profile() as prof:

        for ob_rref in self.ob_rrefs:

            futs.append(
                rpc_async(
                    ob_rref.owner(),
                    _call_method,
                    args = (Observer.load_data, ob_rref, data, validation)
                )
            )

        with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
            f.write("About to wait!" + "\n")

        for fut in futs:
            fut.wait()

        # print(prof.key_averages().table())


        # trace_file = "/sciclone/home20/hmbaier/test_rpc/trace_nproc33.json"
        # Export the trace.
        # prof.export_chrome_trace(trace_file)
        # logger.debug(f"Wrote trace to {trace_file}")


    def load_data(self, world_size):

        """
        Function to load in the data on the Agent
        TO-DO: THIS DOES NOT CURRENTLY LOAD IN THE WHOLE IMAGE (JUST A TUPLE THAT INCLUDES THE LIST OF PARAMETERS THE 
        OBSERVERS WILL NEED TO SET UP THE ENVIRONMENTS) SO ACTUALLY I TENTATIVELY THINK THIS MIGHT BE A FINE WAY TO DO THIS
        """

        with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
            f.write("In load data!! WORLD SIZE: " + str(world_size) + "\n")
            
        # Load in the data
        data = Dataset(census_file = config.census_path, 
                       lc_file = config.lc_path, 
                       imagery_dir = config.imagery_dir, 
                       lc_dir = config.lc_dir,
                       y_file = config.y_path,
                       split = config.tv_split, 
                       world_size = world_size)

        train_dl = data.train_data
        # val_dl = data.val_data
        self.batch_size = world_size

        with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
            f.write("NUMBER OF TRAINING DATA: " + str(len(train_dl)) + "\n")

        # with open("/sciclone/home20/hmbaier/test_rpc/claw_log.txt", "a") as f:
        #     f.write("NUMBER OF VALIDATION DATA: " + str(len(val_dl)) + "\n")

        return train_dl, None


    def run_episode(self, epoch):

        self.training_losses = []

        futs = []

        # with profiler.profile() as prof:

        for ob_rref in self.ob_rrefs:

            futs.append(
                rpc_async(
                    ob_rref.owner(),
                    _call_method,
                    args = (Observer.step, ob_rref, self.agent_rref)
                )
            )

        # with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
        #     f.write("About to wait!" + "\n")

        for fut in futs:
            fut.wait()

        with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
            f.write("EPOCH " + str(epoch) + " LOSS: " + str(np.mean(np.array(self.training_losses))) + "\n")      

        # self.optimize_agent()

    def optimize_conv(self, glimpse, y, features, coords, lc_im):

        # Send the glimpses through the conv net
        output = self.conv_net_updated(glimpse.clone().detach())

        with open("/sciclone/home20/hmbaier/lc_v2/a_pred_log.txt", "a") as f:
            f.write("PREDICTION: " + str(output) + " YS: " + str(y) + "\n")

        loss = self.conv_criterion(output, torch.tensor([[y]]))
        self.conv_optimizer.zero_grad()
        loss.backward()
        self.conv_optimizer.step()

        self.training_losses.append(loss.item())

        self.update_memory(features, coords, glimpse, output, y, lc_im)




    def select_lc(self, lc_im, features, im, y, muni_id):

        try:

            pred, edited = self.lc_model(lc_im, features, im, muni_id)
            loss = self.criterion(pred, y)

            with open("/sciclone/home20/hmbaier/lc_v2/a_pred_log.txt", "a") as f:
                f.write("PREDICTION: " + str(pred.item()) + "  Y: " + str(y.item()) + "  LOSS: " + str(loss.item()) + "  MUNI ID: " + str(muni_id) + " " + str(edited) + "\n")

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.training_losses.append(loss.item())

        except Exception as e:

            with open("/sciclone/home20/hmbaier/lc_v2/a_batch_log.txt", "a") as f:
                f.write("BAD MUNI ID: " + str(muni_id) + " " + str(edited) + "\n" + str(e) + "\n")

        # return self.lc_net_updated(lc_im, features)


    def update_memory(self, features, lc_choices, glimpses, output, y, lc_im):

        glimpses = torch.cat(torch.split(glimpses.unsqueeze(0), config.n_glimpses, dim = 1))

        for i in range(len(output)):
            r = calc_reward(y, output[0])
            mem = (features[i], lc_choices[i], glimpses[i], torch.tensor([r]), lc_im)
            self.memory.append(mem)

            with open("/sciclone/home20/hmbaier/lc_v2/a_mem_log.txt", "a") as f:
                f.write("MEMORY: " + str(mem) + "\n")


    def optimize_agent(self):

        if len(self.memory) > config.memory_limit:
            over = len(self.memory) - config.memory_limit
            self.memory = self.memory[over:]

        if len(self.memory) < config.mem_batch_size:
            return 

        for update in range(0, config.mem_batch_size):
    
            transitions = random.sample(list(self.memory), 1)
            batch = Transition(*zip(*transitions))

            # with open("/sciclone/home20/hmbaier/lc_v2/a_batch_log.txt", "a") as f:
            #     f.write("batch: " + str(batch) + "\n")        

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

            census_batch = torch.cat(batch.census_feat)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)
            lcim_batch = torch.cat(batch.lc_im)

            with open("/sciclone/home20/hmbaier/lc_v2/a_batch_log.txt", "a") as f:
                f.write("lcim_batch: " + str(lcim_batch) + "\n" + "census_batch: " + str(census_batch) + "\n")

            with open("/sciclone/home20/hmbaier/lc_v2/a_batch_log.txt", "a") as f:
                f.write("lcim_batch: " + str(lcim_batch.shape) + "\n" + "census_batch: " + str(census_batch.shape) + "\n")

            # lc_net_updated is the model that we optimize every epoch...
            # So here we run the classification images through the updates model (the value stored in memoery have been ran through non-updated model)
            # We do this because in theory our updated model should produce a better pred & therefore higher reward that we can use to calculate loss
            state_action_values = self.lc_net_updated(lcim_batch, census_batch.unsqueeze(0)).gather(1, action_batch)
            
            # with open("/sciclone/home20/hmbaier/lc_v2/a_batch_log.txt", "a") as f:
            #     f.write("state_batch: " + str(lcim_batch) + "\n" + "action_batch: " + str(action_batch) + "\n" + "reward_batch: " + str(reward_batch) + "\n")

            with open("/sciclone/home20/hmbaier/lc_v2/a_batch_log.txt", "a") as f:
                f.write("state_action_values: " + str(state_action_values) + "\n")




@record
def run_worker(rank, world_size):

    r"""
    This is the entry point for all processes. The rank 0 is the agent. All
    other ranks are observers.
    """

    # os.environ['GLOO_SOCKET_IFNAME'] = "ib0"
    os.environ['TP_SOCKET_IFNAME'] = "ib0"

    # dist.init_process_group(backend = "gloo", rank = rank, world_size = args.world_size)

    # Rank 0 is the agent
    if rank == 0:

        # Set up remote protocol on core
        # rpc.init_rpc(AGENT_NAME, rank = rank, world_size = world_size)

        rpc.init_rpc(AGENT_NAME, rank = rank, world_size = world_size, rpc_backend_options = rpc.TensorPipeRpcBackendOptions(_transports=["uv"], rpc_timeout=5000))

        with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
            f.write("AGENT RPC INITIALIZED!" + "\n")        

        # Initialize the agent in rank 0
        agent = Agent(world_size, config)

        for epoch in range(0, config.num_epochs):

            agent.run_episode(epoch)

        # # i_episode? I think this is equivalent to epochs
        # for i_episode in range(10):

        #     # n_steps = int(TOTAL_EPISODE_STEP / (int(os.environ['WORLD_SIZE']) - 1))

        #     # Run epsiode is basically 1 call of 'train'
        #     agent.run_episode(n_steps = 1, validation = False)
        #     agent.optimize_model()
        #     train_mae = calc_mae(agent.preds_record.values())

        #     agent.run_episode(n_steps = 1, validation = True)
        #     val_mae = calc_mae(agent.preds_record.values())

        #     with open("/sciclone/home20/hmbaier/test_rpc/claw_pred_log.txt", "a") as f:
        #         f.write("EPOCH " + str(i_episode) + "\n" + "Training MAE: " + str(train_mae) + "\n" + "Validation MAE: " + str(val_mae) + "\n")
                           


    # All other ranks are observers who are passively waiting for instructions from agents
    else:

        # Set up remote protocol on core
        rpc.init_rpc(OBSERVER_NAME.format(rank), rank = rank, world_size = world_size, rpc_backend_options = rpc.TensorPipeRpcBackendOptions(_transports=["uv"], rpc_timeout=5000))

        with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
            f.write("OBSERVER RPC INITIALIZED!" + "\n")            

    rpc.shutdown()



def main():

    run_worker(dist.get_rank(), dist.get_world_size())


if __name__ == "__main__":

    os.environ["OMP_NUM_THREADS"] = '24'

    print("here!")

    torch.autograd.set_detect_anomaly(True)

    # main()

    # os.environ['OMP_NUM_THREADS'] = '24'

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


    main()



