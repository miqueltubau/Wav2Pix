import torch.nn as nn
from models.auxiliary_classifier import auxclassifier
#from models.segan.segan_discriminator import Discriminator
from models.segan import Discriminator
from models.spectral_norm import SpectralNorm

class generator(nn.Module):
    def __init__(self, image_size, audio_samples):
        super(generator, self).__init__()

        # defining some useful variables
        self.audio_samples = audio_samples
        self.num_channels = 3
        self.latent_dim = 128
        self.ngf = 64
        self.image_size = image_size

        # defining segan's D
        self.d_fmaps = [16, 32, 128, 256, 512, 1024]
        self.audio_embedding = Discriminator(1, self.d_fmaps, 15, nn.LeakyReLU(0.3), self.audio_samples)
        # defining the auxiliary classifier
        self.aux_classifier = auxclassifier()

        # common network for both architectures when generating 64x64 or 128x18 images
        self.netG = nn.Sequential(
            # state size. (ngf*4) x 8 x 8
            SpectralNorm(nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False)),
            nn.Dropout(),
            # nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            SpectralNorm(nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False)),
            # nn.BatchNorm2d(self.ngf),
            nn.Dropout(),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            SpectralNorm(nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False)),
            nn.Dropout(),
            nn.ReLU(True),
            # If we add here Dropout, we would only generate noise, but not realistic faces
            SpectralNorm(nn.ConvTranspose2d(self.ngf, self.num_channels, 4, 2, 1, bias=False)),
            # state size. (num_channels) x 128 x 128
            nn.Tanh()
        )

        # if we want to generate 64x64 images:
        if self.image_size == 64:
            self.netG = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(self.latent_dim, self.ngf*8, 4, 1, 0, bias=False)),
            nn.Dropout(),
            # nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            self.netG
            )

        # if we want to generate 128 x 128 images:
        if self.image_size == 128:
            self.netG = nn.Sequential(
                SpectralNorm(nn.ConvTranspose2d(self.latent_dim, self.ngf*16, 4, 1, 0, bias=False)),
                nn.Dropout(),
                nn.ReLU(True),
                SpectralNorm(nn.ConvTranspose2d(self.ngf*16, self.ngf*8, 4, 2, 1, bias=False)),
                nn.Dropout(),
                # nn.BatchNorm2d(self.ngf * 4),
                nn.ReLU(True),
                self.netG
            )


    def forward(self, raw_wav):

        # feeding the audio to segan's D
        y, wav_embedding = self.audio_embedding(raw_wav.unsqueeze(1))

        # storing scores after feeding the audio embedding to the classifier (softmax)
        softmax_scores = self.aux_classifier(y)

        # feeding the audio embedding to the GAN generator
        z_vector = y.unsqueeze(2).unsqueeze(3)
        output = self.netG(z_vector)

        return output, z_vector, softmax_scores

