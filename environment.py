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
        self.features = torch.tensor([features])
        self.impath = impath
        self.lcpath = lcpath
        self.im = cv2.imread(self.impath)
        self.H, self.W, self.C = self.im.shape
        # self.lc = torch.from_numpy(cv2.imread(self.lcpath))
        self.to_tens = transforms.ToTensor()
        self.lc = self.to_tens(cv2.imread(self.lcpath))[0].unsqueeze(0).unsqueeze(0)

    def get_features(self):

        """
        Returns the tensor form of the LC + Census features to the training 
        loop so that it can be passed through DQN to select the LC classes 
        to sample from
        """
        
        return torch.tensor(self.features).unsqueeze(0)

    # def sample_glimpses(self, lc_choices):

    #     """
    #     Will randomly sample the pixel coordinates 
    #     to grab glimpses around in the image 
    #     """

    #     coords = []
    #     for lc_type in lc_choices:
    #         indices = (self.lc == lc_type).nonzero(as_tuple=True)
    #         if len(indices[0]) == 0:
    #             coords.append(None)
    #         else:

    #             # if self.id in [3, 9, 11]:
    #             # with open("/sciclone/home20/hmbaier/lc/alog.txt", "a") as f:
    #             #     f.write("NUMBER OF MATCHING INDICES: " + str(self.id) + str(len(indices)) + "\n")

    #             selected_index = random.randint(0, len(indices[0]) - 1)
    #             coord = (int(indices[0][selected_index]), int(indices[1][selected_index]))
    #             coords.append(coord)

    #     return coords

    # def extract_glimpses(self, lc_choices):

    #     """
    #     Extract the imagery at each sampled coordiante and convert 
    #     to tensor form to send through the final prediction model
    #     """

    #     glimpse_coords = self.sample_glimpses(lc_choices)
    #     glimpses = []
    #     for coord in glimpse_coords:
    #         if coord is None:
    #             g = torch.zeros(1, 3, 64, 64)
    #         else:

    #             x_coord, y_coord = coord[0], coord[1]

    #             # if self.id in [3, 9, 11]:
    #             #     with open("/sciclone/home20/hmbaier/lc/alog.txt", "a") as f:
    #             #         f.write("G COORDS IN ENV: " + str(self.id) + str(x_coord) + ", " + str(y_coord) + str(self.im.shape) + "\n")
                        
    #             if (x_coord - self.radius) < 0:
    #                 x_coord += abs(x_coord - self.radius)
    #             elif (x_coord + self.radius) > self.im.shape[0]:
    #                 subtract = self.im.shape[0] - (x_coord + self.radius)
    #                 x_coord -= abs(subtract)

    #             if (y_coord - self.radius) < 0:
    #                 y_coord += abs(y_coord - self.radius)
    #             elif (y_coord + self.radius) > self.im.shape[1]:
    #                 subtract = self.im.shape[1] - (y_coord + self.radius)
    #                 y_coord -= abs(subtract)

    #             g = self.to_tens(self.im[x_coord - self.radius:x_coord + self.radius, 
    #                         y_coord - self.radius:y_coord + self.radius, 
    #                         :]).unsqueeze(0)

    #         # if self.id in [3, 9, 11]:
    #         #     with open("/sciclone/home20/hmbaier/lc/alog.txt", "a") as f:
    #         #         f.write("G SHAPE IN ENV: " + str(self.id) + ": " + str(g.shape) + "\n")

    #         glimpses.append(g)

    #     return torch.cat(glimpses)


    def extract_glimpses(self, coords):

        """
        Extract the imagery at each sampled coordiante and convert 
        to tensor form to send through the final prediction model
        """

        # glimpses = []
        # for coord in glimpse_coords:
        # if coord is None:
        #     g = torch.zeros(1, 3, 64, 64)
        # else:

        edited = False

        x_coord, y_coord = coords[0]
        x_coord, y_coord = x_coord.item(), y_coord.item()

        # if self.id in [3, 9, 11]:
        #     with open("/sciclone/home20/hmbaier/lc/alog.txt", "a") as f:
        #         f.write("G COORDS IN ENV: " + str(self.id) + str(x_coord) + ", " + str(y_coord) + str(self.im.shape) + "\n")
                
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

        g = self.to_tens(self.im[x_coord - self.radius:x_coord + self.radius, 
                    y_coord - self.radius:y_coord + self.radius, 
                    :]).unsqueeze(0)

        # with open("/sciclone/home20/hmbaier/lc/alog.txt", "a") as f:
        #     f.write("G SHAPE IN ENV: " + str(self.id) + ": " + str(g.shape) + "\n")

        return [g, edited]


    def denormalize(self, coords):
        
        """
        Convert coordinates in the range [-1, 1] to
        coordinates in the range [0, T] where `T` is
        the size of the image.
        """

        coords = torch.clamp(coords, -1, 1)
        
        x, y = coords[0]

        X = int(0.5 * (x + 1) * self.W)
        Y = int(0.5 * (y + 1) * self.H)

        return torch.tensor([[X, Y]], dtype = torch.long)


    def get_glimpses(self, coords):

        """
        Returns the stacked gimpse tensors to the training loop 
        so they can be sent through the final prediction model
        """

        return self.extract_glimpses(self.denormalize(coords))