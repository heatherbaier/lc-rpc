from torch.distributed.elastic.multiprocessing.errors import record
import random
import json
import os

from config import get_config

config, _ = get_config()


def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))


class Dataset():

    def __init__(self, census_file, lc_file, imagery_dir, lc_dir, y_file, split, world_size):
        self.imagery_dir = imagery_dir  
        self.world_size = world_size
        self.lc_dir = lc_dir
        self.y_file = y_file
        self.split = split
        self.data = []

        with open(census_file, "r") as f:
            self.census_feats = json.load(f) 

        with open(lc_file, "r") as f:
            self.lc_feats = json.load(f)

        with open(y_file, "r") as f:
            self.ys = json.load(f)

        self.load_data()
        self.batch_size = int(len(self.data) / (world_size - 2))
        self.train_val_split()


    def load_data(self):

        # with open(config.log_name, "a") as f:
        #     f.write("LOADING DATA \n")        

        for im in os.listdir(self.imagery_dir)[8:]:

            try:

                muni_id = im.split(".")[0]
                lcpath = os.path.join(self.lc_dir, str(muni_id) + ".discrete_classification.tif")
                census = list(self.census_feats[muni_id])
                lc = list(self.lc_feats[muni_id])
                # feats = lc + census
                feats = census
                self.data.append([muni_id, feats, os.path.join(self.imagery_dir, im), lcpath, self.ys[muni_id]])

            except Exception as e:

                # with open(config.log_name, "a") as f:
                #     f.write("EXCEPTION: " + str(e) + "\n")   
                
                pass

            if len(self.data) == 48 * 3:

                break

    def split_data(self):

        n = self.batch_size
        self.train_data = [self.data[i:i + n] for i in range(0, len(self.data), n)]


    def train_val_split(self):

        # print("LENGTH OF DATA: ", len(self.data))

        train_num = int(len(self.data) * self.split)
        self.train_num = train_num
        train_indices = random.sample(range(len(self.data)), train_num)
        val_indices = [i for i in range(len(self.data)) if i not in train_indices]
        train_data = [self.data[i] for i in train_indices]
        val_data = [self.data[i] for i in val_indices]   

        self.train_batch_size = int(len(train_data) / (self.world_size - 1))

        if len(val_data) > self.world_size:
            self.val_batch_size = int(len(val_data) / (self.world_size - 1))
            self.val_data = [val_data[i:i + self.val_batch_size] for i in range(0, len(val_data), self.val_batch_size)]
            # with open(config.log_name, "a") as f:
            #     f.write("VAL BATCH SIZE: " + str(self.val_batch_size) + "\n")  
        else:
            self.val_data = [[i] for i in val_data]

        self.train_data = [train_data[i:i + self.train_batch_size] for i in range(0, len(train_data), self.train_batch_size)]

        with open(config.log_name, "a") as f:
            f.write("TRAIN NUM: " + str(self.train_num) + "\n")    

        # with open(config.log_name, "a") as f:
        #     f.write("VAL BATCH SIZE: " + str(len(val_data)) + "\n") 



from config import get_config

config, _ = get_config()

@record
def load_data(world_size):

    """
    Function to load in the data on the Agent
    TO-DO: THIS DOES NOT CURRENTLY LOAD IN THE WHOLE IMAGE (JUST A TUPLE THAT INCLUDES THE LIST OF PARAMETERS THE 
    OBSERVERS WILL NEED TO SET UP THE ENVIRONMENTS) SO ACTUALLY I TENTATIVELY THINK THIS MIGHT BE A FINE WAY TO DO THIS
    """
  
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
    batch_size = world_size

    # with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
    #     f.write("NUMBER OF TRAINING DATA: " + str(len(train_dl)) + "\n")

    # with open("/sciclone/home20/hmbaier/test_rpc/claw_log.txt", "a") as f:
    #     f.write("NUMBER OF VALIDATION DATA: " + str(len(val_dl)) + "\n")

    return data