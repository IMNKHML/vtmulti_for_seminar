import torch
import torch.nn as nn
import sys
sys.path.append('/home/murata-lab/work/imano/vtmulti3/src/models')
from vtmulti3.src.models.Tactile_AE import Tac_Encoder
from data.build_features import change_brightness, add_noise

class TactileEncoder(nn.Module):
    """ Encoder + reparameterization

    self.sequenceでは
    (80, 80, 3) -> (10, 10, 32) -> 64 -> 8

    """
    def __init__(self, device, latent_dim=4, middle_layer_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device

        self.sequence = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8), nn.Mish(),
            nn.AvgPool2d(2, stride=2), 
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16), nn.Mish(),
            nn.AvgPool2d(2, stride=2), 
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32), nn.Mish(),
            # nn.Dropout(0.25), 
            nn.Flatten(),
            nn.Linear(10 * 10 * 32, middle_layer_dim), nn.Mish(),
            
        )

        self.fc_for_mu = nn.Linear(middle_layer_dim, latent_dim)
        self.fc_for_logvar = nn.Linear(middle_layer_dim, latent_dim)

        self.to(device)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std.to(self.device)
        return z
    
    def forward(self, x):
        batch_size = x.size()[0]
        x = x.view(-1, *x.size()[2:])
        out = self.sequence(x)
        mu = self.fc_for_mu(out)
        logvar = self.fc_for_logvar(out)
        z = self.reparameterize(mu, logvar)
        z = z.view(batch_size, -1, self.latent_dim)
        
        return z, mu, logvar

    
class TactileDecoder(nn.Module):

    def __init__(self, device, latent_dim=4, middle_layer_dim=64):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Linear(latent_dim, middle_layer_dim),
            nn.Mish(),
            nn.Linear(middle_layer_dim, 10 * 10 * 32),
            nn.Mish(),
            _Lambda(lambda x: x.view(-1, 32, 10, 10)),
            nn.BatchNorm2d(32), nn.Mish(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16), nn.Mish(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8), nn.Mish(),
            nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(3), nn.Sigmoid(),
        )

        self.to(device)

    def forward(self, z):
        return self.sequence(z)


class TactileVAE(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.encoder = TactileEncoder(device=device, latent_dim=4, middle_layer_dim=64)
        self.decoder = TactileDecoder(device=device, latent_dim=4, middle_layer_dim=64)

        self.to(device)

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        y = self.decoder(z)

        return y, mu, logvar

class _Lambda(torch.nn.Module):
    """

    nn.Sequential内で強引にtensor.viewを使うためのクラス(綺麗にしたいから)

    Example:
        self.sequential = nn.Sequential(
            nn.Linear(10000, 1000),
            Lambda(lambda x: x.view([-1, 10, 10, 10])),
            nn.Conv2d( ...
        )

    """

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)
