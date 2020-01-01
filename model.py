import torch
import torch.nn as nn


class ResLayer(nn.Module):
    def __init__(self, in_channels, out_channels, k, p):
        super(ResLayer, self).__init__()

        self.net = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=k,
                                           padding=p),  # Output tensor will have shape (Batch_Size x 64 x W x H)
                                 nn.BatchNorm2d(out_channels),
                                 nn.PReLU(),
                                 nn.Conv2d(out_channels, out_channels,
                                           kernel_size=k, padding=p),
                                 nn.BatchNorm2d(out_channels))

    def forward(self, x):
        return x + self.net(x)


class UpscaleLayer(nn.Module):
    def __init__(self, in_channels, scaleFactor, k, p):
        super(UpscaleLayer, self).__init__()

        # Here out_channels = square of scaleFactor multiplied by in_channels
        # Since in_channels to this layer is 64 and scaleFactor is 2 then out_channels becomes 256
        # nn.PixelShuffle(scaleFactor) divides out_channels with square of scaleFactor and multiplies W & H with scaleFactor
        self.net = nn.Sequential(nn.Conv2d(in_channels,
                                           in_channels * (scaleFactor ** 2),
                                           kernel_size=k,
                                           padding=p),  # Output tensor will have shape (Batch_Size x 256 x W x H)
                                 nn.PixelShuffle(scaleFactor),
                                 # Output tensor will have shape (Batch_Size x 64 x 2W x 2H)
                                 nn.PReLU())

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, number_residual):
        super(Generator, self).__init__()

        self.n_residual = number_residual

        '''                          
        Since formulae for calculating shape of output of a convulational layer 
        is :- ((i-k) + 2p +1)/s, where "i" could be width or height of input,
        "k" is kernel size, "p" is padding value, "s" is stride value.
    
        Consider input tensor shape is (Batch_Size x 3 x W x H)
        '''

        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=9, padding=4),
                                   nn.PReLU())
        # Output tensor will have shape (Batch_Size x 64 x W x H)

        for i in range(self.n_residual):
            # ResLayer(in_channels, out_channels, kernel_size, padding)
            self.add_module('residual' + str(i + 1), ResLayer(64, 64, 3, 1))
            # Output tensor will have shape (Batch_Size x 64 x W x H)

        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
                                   nn.PReLU())
        # Output tensor will have shape (Batch_Size x 64 x W x H)

        # UpscaleLayer(in_channels, scaleFactor, kernel_size, padding)
        self.upscaler = nn.Sequential(UpscaleLayer(64, 2, 3, 1),
                                      # Output tensor will have shape (Batch_Size x 64 x 2W x 2H)
                                      UpscaleLayer(64, 2, 3, 1),
                                      # Output tensor will have shape (Batch_Size x 64 x 4W x 4H)
                                      nn.Conv2d(64, 3, kernel_size=9, padding=4))
        # Output tensor will have shape (Batch_Size x 3 x 4W x 4H)
        # Thus by here we have 4 time scaled resolution of our input image.

    def forward(self, x):
        out = self.conv1(x)

        for i in range(self.n_residual):
            out = self.__getattr__('residual' + str(i + 1))(out)

        out = self.conv2(out)

        out = self.upscaler(out)
        return torch.tanh(out)


class Descriminator(nn.Module):
    def __init__(self, leakyFactor=0.15):
        super(Descriminator, self).__init__()

        self.net = nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),
                                 nn.LeakyReLU(leakyFactor),
                                 # Output tensor will have shape (Batch_Size x 64 x W x H) here W & H are dimensions of images scaled by Generator

                                 nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                                 nn.BatchNorm2d(64),
                                 nn.LeakyReLU(leakyFactor),
                                 # Output tensor will have shape (Batch_Size x 64 x W/2 x H/2)

                                 nn.Conv2d(64, 128, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(128),
                                 nn.LeakyReLU(leakyFactor),
                                 # Output tensor will have shape (Batch_Size x 128 x W x H)
                                 # Here W = W of previous layer. Same is with H.

                                 nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
                                 nn.BatchNorm2d(128),
                                 nn.LeakyReLU(leakyFactor),
                                 # Output tensor will have shape (Batch_Size x 128 x W/2 x H/2)
                                 # By here total we have downscaled W & H by factor of 4 and upscaled channels from 3 to 128.

                                 nn.Conv2d(128, 256, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(256),
                                 nn.LeakyReLU(leakyFactor),
                                 # Output tensor will have shape (Batch_Size x 256 x W x H)
                                 # Here W = W of previous layer. Same is with H.
                                 # By here total we have downscaled W & H by factor of 4 and upscaled channels from 3 to 256.

                                 nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                                 nn.BatchNorm2d(256),
                                 nn.LeakyReLU(leakyFactor),
                                 # Output tensor will have shape (Batch_Size x 256 x W/2 x H/2)
                                 # By here total we have downscaled W & H by factor of 8 and upscaled channels from 3 to 256.

                                 nn.Conv2d(256, 512, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(512),
                                 nn.LeakyReLU(leakyFactor),
                                 # Output tensor will have shape (Batch_Size x 512 x W x H)
                                 # Here W = W of previous layer. Same is with H.
                                 # By here total we have downscaled W & H by factor of 8 and upscaled channels from 3 to 512.

                                 nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
                                 nn.BatchNorm2d(512),
                                 nn.LeakyReLU(leakyFactor),
                                 # Output tensor will have shape (Batch_Size x 512 x W/2 x H/2)
                                 # By here total we have downscaled W & H by factor of 16 and upscaled channels from 3 to 512.

                                 nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(512, 1024, kernel_size=1),
                                 # Output tensor will have shape (Batch_Size x 1024 x W x H)
                                 # Here W = W of previous conv layer. Same is with H.

                                 nn.LeakyReLU(leakyFactor),
                                 nn.Conv2d(1024, 1, kernel_size=1),
                                 # Output tensor will have shape (Batch_Size x 1 x W x H)
                                 # Here W = W of previous conv layer. Same is with H.
                                 )

    def forward(self, x):
        out = self.net(x)
        return torch.sigmoid(out)





