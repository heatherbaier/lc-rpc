from torchvision import models
import torch

from config import *

config, _ = get_config()

class GlimpseNetwork(torch.nn.Module):
    
    def __init__(self, resnet, hidden_size):
        super().__init__()
        
        self.radius = 64

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = torch.nn.ReLU(inplace = False)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = torch.nn.Linear(512, hidden_size)
        
    def denormalize(self, coords, im):
        
        """
        Convert coordinates in the range [-1, 1] to
        coordinates in the range [0, T] where `T` is
        the size of the image.
        """
        
        self.B, self.C, self.H, self.W = im.shape

        coords = torch.clamp(coords, -1, 1)
        
        x, y = coords[0]

        X = int(0.5 * (x + 1) * self.W)
        Y = int(0.5 * (y + 1) * self.H)

        return torch.tensor([[X, Y]], dtype = torch.long)
    
    def extract_glimpse(self, im, coords):
        
        """
        Extract the imagery at each sampled coordiante and convert 
        to tensor form to send through the final prediction model
        """
        
        self.im = im
        
        coords = self.denormalize(coords, im)
        
#         print(coords)
        
        x_coord, y_coord = coords[0]
        x_coord, y_coord = x_coord.item(), y_coord.item()

        if (x_coord - self.radius) < 0:
            x_coord += abs(x_coord - self.radius)

        elif (x_coord + self.radius) > self.H:
            subtract = self.H - (x_coord + self.radius)
            x_coord -= abs(subtract)

        if (y_coord - self.radius) < 0:
            y_coord += abs(y_coord - self.radius)

        elif (y_coord + self.radius) > self.W:
            subtract = self.W - (y_coord + self.radius)
            y_coord -= abs(subtract)
            
        x_coord, y_coord = int(x_coord), int(y_coord)
                        
        g = self.im[:, :, x_coord - self.radius:x_coord + self.radius, 
                    y_coord - self.radius:y_coord + self.radius]
        
        return g
    
    def forward(self, x, coord):
        
        """
        x: 4D tensor of whole Landsat image
        coord: Tensor with 2 elements that represent normalized coordinates in image
        """
        
        glimpse = self.extract_glimpse(x, coord)
        
        x = self.conv1(glimpse)
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
        
        return x
    
    
class CoreNetwork(torch.nn.Module):
    
    """The core network.
    An RNN that maintains an internal state by integrating
    information extracted from the history of past observations.
    It encodes the agent's knowledge of the environment through
    a state vector `h_t` that gets updated at every time step `t`.
    Concretely, it takes the glimpse representation `g_t` as input,
    and combines it with its internal state `h_t_prev` at the previous
    time step, to produce the new internal state `h_t` at the current
    time step.
    In other words:
        `h_t = relu( fc(h_t_prev) + fc(g_t) )`
    Args:
        input_size: input size of the rnn.
        hidden_size: hidden size of the rnn.
        g_t: a 2D tensor of shape (B, hidden_size). The glimpse
            representation returned by the glimpse network for the
            current timestep `t`.
        h_t_prev: a 2D tensor of shape (B, hidden_size). The
            hidden state vector for the previous timestep `t-1`.
    Returns:
        h_t: a 2D tensor of shape (B, hidden_size). The hidden
            state vector for the current timestep `t`.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.i2h = torch.nn.Linear(input_size, hidden_size)
        self.h2h = torch.nn.Linear(hidden_size, hidden_size)
        
        self.relu = torch.nn.ReLU()

    def forward(self, g_t, h_t_prev):
        
        h1 = self.i2h(g_t)
        h2 = self.h2h(h_t_prev)
        h_t = self.relu(h1 + h2)   
        return h_t
    
    
class lcConv(torch.nn.Module):
    
    def __init__(self, resnet, fc_size):
        super().__init__()
    
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = resnet.bn1
        self.relu = torch.nn.ReLU(inplace = False)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self.fc = torch.nn.Linear(512, fc_size)   

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
        return x

class LocationNetwork(torch.nn.Module):
    
    """The location network.
    Uses the internal state `h_t` of the core network to
    produce the location coordinates `l_t` for the next
    time step.
    Concretely, feeds the hidden state `h_t` through a fc
    layer followed by a tanh to clamp the output beween
    [-1, 1]. This produces a 2D vector of means used to
    parametrize a two-component Gaussian with a fixed
    variance from which the location coordinates `l_t`
    for the next time step are sampled.
    Hence, the location `l_t` is chosen stochastically
    from a distribution conditioned on an affine
    transformation of the hidden state vector `h_t`.
    Args:
        input_size: input size of the fc layer.
        output_size: output size of the fc layer.
        std: standard deviation of the normal distribution.
        h_t: the hidden state vector of the core network for
            the current time step `t`.
    Returns:
        mu: a 2D vector of shape (B, 2).
        l_t: a 2D vector of shape (B, 2).
    """

    def __init__(self, input_size, resnet, fc_size):
        super().__init__()
        
#         self.linear1 = torch.nn.Linear(128 + 94, 256)
        self.linear1 = torch.nn.Linear(224, 256)
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128, 64)
        self.linear4 = torch.nn.Linear(64, 2)
        self.relu = torch.nn.ReLU()
        
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = resnet.bn1
#         self.relu_resnet = torch.nn.ReLU(inplace = False)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
#         self.fc = torch.nn.Linear(512, fc_size)           
        self.fc = torch.nn.Linear(512, 2)           

    def forward(self, h_t, lc_im, census):
        
        x = self.conv1(lc_im)
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
        
#         x = torch.cat((h_t, census), dim = 1)
        x = torch.cat((h_t, x, census), dim = 1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        return x
        

class ActionNetwork(torch.nn.Module):
    
    def __init__(self, input_size):
        super().__init__()
        
        self.linear1 = torch.nn.Linear(input_size + 2, 256)
        self.linear2 = torch.nn.Linear(256, 128)
        self.linear3 = torch.nn.Linear(128, 64)
        self.linear4 = torch.nn.Linear(64, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, coords, h_t, census):    
        x = torch.cat((coords, h_t), dim = 1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        return x
    
    
class lcNet(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.gnet = GlimpseNetwork(resnet = models.resnet18(),
                             hidden_size = config.glimpse_hidden_size)
        self.core_net = CoreNetwork(input_size = config.glimpse_hidden_size,
                               hidden_size = config.core_hidden_size)
#         self.lc_net = lcConv(resnet = models.resnet18(),
#                        fc_size = config.lc_fc_size)
        self.loc_net = LocationNetwork(input_size = config.core_hidden_size + config.lc_fc_size + 94, 
                                       resnet = models.resnet18(),
                                       fc_size = config.lc_fc_size)
        self.action_net = ActionNetwork(input_size = config.core_hidden_size)
        
    def forward(self, x):
        
        im, coords, h_t, lc_im, census, last = x
        
        glimpse = self.gnet(im, coords)
        h_t = self.core_net(glimpse, h_t)
#         lc_fc = self.lc_net(lc_im)
        coords = self.loc_net(h_t, lc_im, census)
        
        if last == 'True':
            
            return self.action_net(coords, h_t, census)
            
        return coords, h_t