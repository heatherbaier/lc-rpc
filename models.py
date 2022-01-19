from torch import nn
import torch

class lcNet(nn.Module):

    def __init__(self, resnet):

        super(lcNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = resnet.bn1
        self.relu = torch.nn.ReLU(inplace = False)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = torch.nn.Linear(512, 2)
        # self.fc = torch.nn.Linear(606, 2)

    def forward(self, x, census):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(start_dim = 1)
        # x = torch.cat((x.clone(), census), dim = 1)
        x = self.fc(x)
        return x
    
    
class ConvNet(nn.Module):

    def __init__(self, resnet, n_glimpses):

        super(ConvNet, self).__init__()
        self.n_glimpses = n_glimpses
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = torch.nn.ReLU(inplace = False)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = torch.nn.Linear(512, 1)
        # self.fc2 = torch.nn.Linear(8 * self.n_glimpses, 1)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.flatten(start_dim = 1)
        x = self.fc(x)
        # print(x.shape)
        # x = self.fc(x).unsqueeze(0)
        # x = torch.cat(torch.split(x, self.n_glimpses, dim = 1)).flatten(start_dim = 1)
        # return self.fc2(x)
        return x
    
    
from torchvision import transforms, models
import random
import torch
import json
import cv2 


class modelEnv(nn.Module):

    def __init__(self):
        super(modelEnv, self).__init__()
        self.radius = 64
        self.to_tens = transforms.ToTensor()

    def extract_glimpses(self, coords, im, muni_id):

        """
        Extract the imagery at each sampled coordiante and convert 
        to tensor form to send through the final prediction model
        """

        with open("/sciclone/home20/hmbaier/lc_v2/a_coords_log.txt", "a") as f:
            f.write("COORDS FOR: " + str(muni_id) + " " + str(coords) + "\n")

        edited = False
        
        self.im = im

        x_coord, y_coord = coords[0]
        x_coord, y_coord = x_coord.item(), y_coord.item()

        if (x_coord - self.radius) < 0:
            x_coord += abs(x_coord - self.radius)
            edited = True

        elif (x_coord + self.radius) > self.H:
            subtract = self.H - (x_coord + self.radius)
            x_coord -= abs(subtract)
            edited = True

        if (y_coord - self.radius) < 0:
            y_coord += abs(y_coord - self.radius)
            edited = True

        elif (y_coord + self.radius) > self.W:
            subtract = self.W - (y_coord + self.radius)
            y_coord -= abs(subtract)
            edited = True

        g = torch.tensor(self.to_tens(self.im[x_coord - self.radius:x_coord + self.radius, 
                    y_coord - self.radius:y_coord + self.radius, 
                    :]).unsqueeze(0), dtype = torch.float32)
        
        return [g, edited]

    def denormalize(self, coords, dims):
        
        """
        Convert coordinates in the range [-1, 1] to
        coordinates in the range [0, T] where `T` is
        the size of the image.
        """
        
        self.B, self.C, self.H, self.W = dims

        coords = torch.clamp(coords, -1, 1)
        
        x, y = coords[0]

        X = int(0.5 * (x + 1) * self.W)
        Y = int(0.5 * (y + 1) * self.H)

        return torch.tensor([[X, Y]], dtype = torch.long)

    def forward(self, coords, dims, im, muni_id):

        """
        Returns the stacked gimpse tensors to the training loop 
        so they can be sent through the final prediction model
        """

        return self.extract_glimpses(self.denormalize(coords, dims), im, muni_id)


class lcModel(nn.Module):
    
    def __init__(self):
        
        super(lcModel, self).__init__()
        
        self.get_coords = lcNet(models.resnet18())
        self.glimpse_net = modelEnv()
        self.conv_net = ConvNet(models.resnet18(), 5)
        
    def forward(self, lc_im, census, im, muni_id):
        
        coords = self.get_coords(lc_im, census)
        glimpse, edited = self.glimpse_net(coords, lc_im.shape, im, muni_id)
#         print(glimpse.shape)
        return self.conv_net(glimpse), edited