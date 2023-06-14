
import torch
import torch.nn as nn
import numpy as np

#ToDO Fill in the __ values
class Smoother(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()

        self.cross_entropy = nn.CrossEntropyLoss()

        self.latent_dim = config.latent_dim
        self.kl_weight = config.kl_weight

        modules = []
        self.hidden_dims = hidden_dims = config.hidden_config
        pooling = np.ones(len(self.hidden_dims)) * 2

        # Build Encoder
        in_channels = 3
        current_size = config.image_size
        for h_dim, pool in zip(hidden_dims, pooling):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size = 3 , stride = 1, padding = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.MaxPool2d((pool,pool)),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
            current_size = current_size // pool

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * current_size, self.latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * current_size, self.latent_dim)

        self.decoder = nn.Linear(self.latent_dim, num_classes)

    def encode(self, input):

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components of the latent Gaussian distribution
        print(self.fc_mu)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        result = self.decoder(z)
        return result

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        output = self.decode(z)
        self.mu, self.log_var = mu, log_var
        return  output

    def loss_function(self, output, labels):

        label_loss = self.cross_entropy(output, labels)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + self.log_var - self.mu ** 2 - self.log_var.exp(), dim = 1), dim = 0) # Analytic KL Divergence Loss from isotropic gaussian

        loss = label_loss + self.kl_weight * kld_loss

        return loss
