import json
import os


def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))

class Dataset():

    def __init__(self, census_file, lc_file, imagery_dir, lc_dir, y_file, split, world_size):

        self.imagery_dir = imagery_dir  
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
        self.batch_size = int(len(self.data) / world_size)
        self.split_data()


    def load_data(self):

        for im in os.listdir(self.imagery_dir)[8:]:

            try:

                muni_id = im.split(".")[0]
                lcpath = os.path.join(self.lc_dir, str(muni_id) + ".discrete_classification.tif")
                census = list(self.census_feats[muni_id])
                lc = list(self.lc_feats[muni_id])
                # feats = lc + census
                feats = census
                self.data.append([muni_id, feats, os.path.join(self.imagery_dir, im), lcpath, self.ys[muni_id]])

            except:

                pass

            if len(self.data) == 24:

                break

    def split_data(self):

        n = self.batch_size
        self.train_data = [self.data[i:i + n] for i in range(0, len(self.data), n)]