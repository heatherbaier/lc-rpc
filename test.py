import json
import os

from config import get_config

config, _ = get_config()


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
            f.write("IN DATALOADER 1 " + str(census_file) + "\n")        

        # try:
        with open(census_file, "r") as f:
            self.census_feats = json.load(f)
        # except Exception as e:
        #     with open("/sciclone/home20/hmbaier/lc_v2/alog.txt", "a") as f:
        #         f.write("ERROR MESSAGE: " + str(e) + "\n")       

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



data = Dataset(census_file = config.census_path, 
                lc_file = config.lc_path, 
                imagery_dir = config.imagery_dir, 
                lc_dir = config.lc_dir,
                y_file = config.y_path,
                split = config.tv_split, 
                world_size = 12)