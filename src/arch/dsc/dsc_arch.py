
import torch
import torch.nn as nn
import numpy as np

#ToDO Fill in the __ values
class Smoother(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()

        self.base_criterion = nn.CrossEntropyLoss()

        self.latent_smoothing = config.latent_smoothing

        self.latent_dim = config.latent_dim
        self.kl_weight = config.kl_weight

        modules = []
        self.hidden_dims = hidden_dims = config.hidden_config
        pooling = np.ones(len(self.hidden_dims), dtype=int) * 2

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

        if not self.latent_smoothing:
            self.direct = nn.Linear(hidden_dims[-1] * current_size, num_classes)
        else:
            self.gaussian_parameters = nn.ModuleDict(
                {
                    'mean':  nn.Linear(hidden_dims[-1] * current_size, self.latent_dim),
                    'variance': nn.Linear(hidden_dims[-1] * current_size, self.latent_dim)
                }
            )

            self.decoder = nn.Linear(self.latent_dim, num_classes)

    def standard_forward(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        result = self.direct(result)
        return result

    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components of the latent Gaussian distribution
        mu = self.gaussian_parameters['mean'](result)
        log_var = self.gaussian_parameters['variance'](result)
        return [mu, log_var]
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        result = self.decoder(z)
        return result

    def forward(self, input):

        if not self.latent_smoothing:
            return self.standard_forward(input)
        
        mu, log_var = self.encode(input)
        self.mu, self.log_var = mu, log_var
        z = self.reparameterize(mu, log_var)

        output = self.decode(z)

        return  output
    
    def register_base_criterion(self, criterion):
        self.base_criterion = criterion

    def loss_function(self, output, labels):
        if not self.latent_smoothing:
            return self.base_criterion(output, labels)

        label_loss = self.base_criterion(output, labels)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + self.log_var - self.mu ** 2 - self.log_var.exp(), dim = 1), dim = 0) # Analytic KL Divergence Loss from isotropic gaussian

        loss = label_loss + self.kl_weight * kld_loss + self.config.beta * torch.sum(torch.norm(self.decoder.weight, dim=1) ** 2)

        return loss
