import torch.nn as nn

class auxclassifier(nn.Module):

    def __init__(self):
        super(auxclassifier,self).__init__()

        self.latent_vector_dim = 128
        self.net = nn.Sequential(
            nn.Linear(in_features= self.latent_vector_dim, out_features=200),
            nn.BatchNorm1d(num_features=200),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=200,out_features=10),
            nn.BatchNorm1d(num_features=10),
        )

    def forward(self,x):
        return self.net(x)