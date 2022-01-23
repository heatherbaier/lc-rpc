from torch.distributed.elastic.multiprocessing.errors import record
import json
import os

from config import get_config

config, _ = get_config()


def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))


class Dataset():

    def __init__(self, census_file, lc_file, imagery_dir, lc_dir, y_file, split, world_size):

        
        with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
            f.write("IN DATALOADER \n")

        self.imagery_dir = imagery_dir  
        self.lc_dir = lc_dir
        self.y_file = y_file
        self.split = split
        self.data = []

        with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
            f.write("IN DATALOADER 1 \n")        

        try:
            with open(census_file, "r") as f:
                self.census_feats = json.load(f)
        except Exception as e:
            with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
                f.write("ERROR MESSAGE: " + str(e) + "\n")       

        with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
            f.write("IN DATALOADER 2 \n")            

        with open(lc_file, "r") as f:
            self.lc_feats = json.load(f)

        with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
            f.write("IN DATALOADER 3 \n")    

        with open(y_file, "r") as f:
            self.ys = json.load(f)

        with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
            f.write("IN DATALOADER 4 \n")    

        self.load_data()
        self.batch_size = int(len(self.data) / world_size)

        # with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
        #     f.write(str(self.data) + "\n")   

        with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
            f.write("ABOUT TO SPLIT DATA \n")   

        self.split_data()

        with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
            f.write("DONE LOADING DATA \n")   


    def load_data(self):

        with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
            f.write("LOADING DATA \n")        

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

                with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
                    f.write("EXCEPTION: " + str(e) + "\n")   
                
                # pass

            if len(self.data) == 48:

                break

    def split_data(self):

        n = self.batch_size
        self.train_data = [self.data[i:i + n] for i in range(0, len(self.data), n)]



from config import get_config

config, _ = get_config()

@record
def load_data(world_size):

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
    batch_size = world_size

    with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
        f.write("NUMBER OF TRAINING DATA: " + str(len(train_dl)) + "\n")

    # with open("/sciclone/home20/hmbaier/test_rpc/claw_log.txt", "a") as f:
    #     f.write("NUMBER OF VALIDATION DATA: " + str(len(val_dl)) + "\n")

    return train_dl, None