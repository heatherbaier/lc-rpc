from torchvision import transforms
import random
import torch
import json
import cv2 


class Env():

    def __init__(self, muni_id, features, y, impath, lcpath, id):

        """
        Initialize the image's training environment
        """

        self.y = torch.tensor([[y]])
        self.id = id
        self.radius = 128
        self.muni_id = muni_id
        self.features = torch.nan_to_num(torch.tensor([features]), nan=0.0)
        self.impath = impath
        self.lcpath = lcpath
        self.im = cv2.imread(self.impath)
        self.H, self.W, self.C = self.im.shape
        self.to_tens = transforms.ToTensor()
        self.lc = self.to_tens(cv2.imread(self.lcpath))[0].unsqueeze(0).unsqueeze(0)
        self.im = self.to_tens(self.im).unsqueeze(0)


