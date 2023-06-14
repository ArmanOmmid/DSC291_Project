
import torch
import torch.nn as nn

#ToDO Fill in the __ values
class Smoother(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()

        self.latent_dim = config.latent_dim
        self.kl_weight = config.kl_weight

        modules = []
        hidden_dims = config.layer_config

        # Build Encoder
        in_channels = 3
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size = 3 , stride = 1, padding = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1] * 4, self.latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * 4, self.latent_dim)

        self.decoder_input = nn.Linear(self.latent_dim, hidden_dims[-1] * 4)
        self.decoder = nn.Linear(self.latent_dim, num_classes)

    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
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

        label_loss = nn.CrossEntropyLoss(output, labels)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + self.log_var - self.mu ** 2 - self.log_var.exp(), dim = 1), dim = 0) # Analytic KL Divergence Loss from isotropic gaussian

        loss = label_loss + self.kl_weight * kld_loss

        return loss
