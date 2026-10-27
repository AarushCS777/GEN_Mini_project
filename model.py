import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1):
        """
        DCGAN Generator for 28x28 grayscale Fashion-MNIST images.
        nz: Size of latent z vector
        ngf: Size of feature maps in generator
        nc: Number of channels in output image (1 for grayscale)
        """
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # Input is Z, going into a convolution
            # Input size: (nz) x 1 x 1
            nn.ConvTranspose2d(nz, ngf * 4, 7, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State size: (ngf*4) x 7 x 7

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State size: (ngf*2) x 14 x 14

            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # Output size: (nc) x 28 x 28
        )

    def forward(self, input_tensor):
        return self.main(input_tensor)


class Discriminator(nn.Module):
    def __init__(self, ndf=64, nc=1):
        """
        DCGAN Discriminator for 28x28 grayscale Fashion-MNIST images.
        ndf: Size of feature maps in discriminator
        nc: Number of channels in input image (1 for grayscale)
        """
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # Input size: (nc) x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf) x 14 x 14

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State size: (ndf*2) x 7 x 7

            nn.Conv2d(ndf * 2, 1, 7, 1, 0, bias=False),
            nn.Sigmoid()
            # Output size: 1 x 1 x 1 (probability)
        )

    def forward(self, input_tensor):
        return self.main(input_tensor).view(-1, 1).squeeze(1)


def weights_init(m):
    """Custom weight initialization according to DCGAN paper."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)