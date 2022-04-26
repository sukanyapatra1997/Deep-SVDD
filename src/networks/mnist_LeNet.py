import torch
import torch.nn as nn
import torch.nn.functional as F

from base.base_net import BaseNet


class MNIST_LeNet(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim = 2*23*76 # 32
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2)
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        # self.fc1 = nn.Linear(4 * 7 * 7, self.rep_dim, bias=False)
        self.fc1 = nn.Linear(4 * 46 * 152, self.rep_dim, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(F.leaky_relu(self.bn1(x)))
        x = self.conv2(x)
        x = self.pool(F.leaky_relu(self.bn2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class MNIST_LeNet_Autoencoder(BaseNet):

    def __init__(self):
        super().__init__()

        self.rep_dim =  2*23*76 #32
        self.pool = nn.MaxPool2d(2, 2)

        
        # Encoder (must match the Deep SVDD network above)
        self.conv1 = nn.Conv2d(1, 8, 5, bias=False, padding=2) 
        self.bn1 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.conv2 = nn.Conv2d(8, 4, 5, bias=False, padding=2)
        self.bn2 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        # self.fc1 = nn.Linear(4 * 7 * 7, self.rep_dim, bias=False)
        self.fc1 = nn.Linear(4 * 46 * 152, self.rep_dim, bias=False)

        # Decoder
        self.deconv1 = nn.ConvTranspose2d(2, 4, 5, bias=False, padding=2)
        self.bn3 = nn.BatchNorm2d(4, eps=1e-04, affine=False)
        self.deconv2 = nn.ConvTranspose2d(4, 8, 5, bias=False, padding=2)
        self.bn4 = nn.BatchNorm2d(8, eps=1e-04, affine=False)
        self.deconv3 = nn.ConvTranspose2d(8, 1, 5, bias=False, padding=2)

    def forward(self, x):
        #print(f'x_type: {type(x)}')
        # input (1,184,608)
        x = self.conv1(x) # (5, 8, 184, 608) mnist (5,8,28,28)
        x = self.pool(F.leaky_relu(self.bn1(x))) #(5, 8, 92, 304) mnist (5,8,14,14)
        x = self.conv2(x) # (5, 4, 92, 304) mnist (5,4,14,14)
        x = self.pool(F.leaky_relu(self.bn2(x))) #(5, 4, 46, 152) mnist (5,4,7,7)
        x = x.view(x.size(0), -1) #(5, 27968) mnist (5,196)
        x = self.fc1(x) #(5,32)
        x = x.view(x.size(0), int(self.rep_dim / (23*76)), 23, 76) #(5,2,23,76)
        x = F.interpolate(F.leaky_relu(x), scale_factor=2) #(5,2,46,152)
        x = self.deconv1(x) #(5,4,46,152)
        x = F.interpolate(F.leaky_relu(self.bn3(x)), scale_factor=2) #(5,4,92,304)
        x = self.deconv2(x) #(5,8,92,304)
        x = F.interpolate(F.leaky_relu(self.bn4(x)), scale_factor=2) #(5,8,184,608)
        x = self.deconv3(x) #(5,1,184,608)
        x = torch.sigmoid(x) #(5,1,184,608)

        return x
