import torch
import torch.nn as nn
from models.spectral_norm import SpectralNorm
from scripts.utils import Concat_embed


class discriminator(nn.Module):
    def __init__(self, image_size):
        super(discriminator, self).__init__()
        self.image_size = image_size
        self.num_channels = 3
        self.latent_space = 128
        self.ndf = 64

        # common network for both architectures, when generating 64x64 or 128x18 images
        self.netD_1 = nn.Sequential(
            # input is (nc) x 64 x 64
            SpectralNorm(nn.Conv2d(self.num_channels, self.ndf, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            SpectralNorm(nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            SpectralNorm(nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            SpectralNorm(nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # if we are feeding D with 64x64 images:
        if self.image_size == 64:
            self.netD_2 = nn.Conv2d(self.ndf * 8 + self.latent_space, 1, 4, 1, 0, bias=False)

        # if we are feeding D with 128x128 images:
        elif self.image_size == 128:
            self.netD_1 = nn.Sequential(
                self.netD_1,
                SpectralNorm(nn.Conv2d(self.ndf * 8, self.ndf * 16, 4, 2, 1, bias=False)),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.netD_2 = nn.Conv2d(self.ndf * 16 + self.latent_space, 1, 4, 1, 0, bias=False)


    def forward(self, input_image, z_vector):

        # feeding  input images to the first stack of conv layers
        x_intermediate = self.netD_1(input_image)

        # replicating the speech embedding spatially and performing a depth concatenation with the embedded audio after
        # being fed into segan's D
        dimensions = list(x_intermediate.shape)
        x = torch.cat([x_intermediate, z_vector.repeat(1,1,dimensions[2],dimensions[3])], 1)

        # feeding to the last conv layer.
        x = self.netD_2(x)

        return x.view(-1, 1).squeeze(1), x_intermediate
