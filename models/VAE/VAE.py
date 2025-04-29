from .base import BaseVAE
import torch
import torch.nn as nn

class VariationalAutoencoder(BaseVAE):
    def __init__(self, 
                 encoder: nn.Module, 
                 decoder: nn.Module, 
                 latent_dim: int):
        super(VariationalAutoencoder, self).__init__(encoder=encoder, decoder=decoder, latent_dim=latent_dim)

    def forward(self, x: torch.Tensor, L: int = 1):
        B = x.size(0)

        # q(z|x)
        x_f = self.encoder(x).flatten(1)
        mu = self.fc_mu(x_f)
        log_var = self.fc_logvar(x_f)

        # z ~ q(z|x)
        z = self.reparametrize(mu=mu, log_var=log_var, L=L)

        # p(z|x)
        z_flat = z.view(B * L, -1)
        x_hat = self.decoder(z_flat)
        x_hat = x_hat.view(B, L, *x.shape[1:])

        return x_hat, z, mu, log_var