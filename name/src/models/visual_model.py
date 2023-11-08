import torch
import torch.nn as nn
from RNN import GRU
from RNN import LSTM
# from NewGate_06 import NewDeepGate_06
from data.build_features import change_brightness, add_noise

class VisionEncoder(nn.Module):
    """ Encoder + reparameterization

    self.sequenceでは
        (32, 32, 3) -> (16, 16, 16) -> 16 -> 8
    になる

    """
    def __init__(self, device, latent_dim=4, middle_layer_dim=16):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device

        self.sequence = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8), nn.Mish(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16), nn.Mish(),
            # nn.Dropout(0.25), 
            nn.Flatten(),
            nn.Linear(16 * 16 * 16 , middle_layer_dim), nn.Mish(),
            
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
    
class VisionDecoder(nn.Module):
    def __init__(self, device, latent_dim=4, middle_layer_dim=16):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Linear(latent_dim, middle_layer_dim),
            nn.Mish(),
            nn.Linear(middle_layer_dim, 16 * 16 * 16),
            nn.Mish(),
            _Lambda(lambda x: x.view(-1, 16, 16, 16)),
            nn.BatchNorm2d(16), nn.Mish(),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(8), nn.Mish(),
            nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(3), nn.Sigmoid(),
        )

        self.to(device)

    def forward(self, z):
        return self.sequence(z)
    
class VisionVAE(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.encoder = VisionEncoder(device=device, latent_dim=4, middle_layer_dim=16)
        self.decoder = VisionDecoder(device=device, latent_dim=4, middle_layer_dim=16)

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
        
