
import torch
import torch.nn as nn
import torchvision
import numpy as np

#ToDO Fill in the __ values
class Smoother(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()

        self.config = config
        self.num_classes = num_classes

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

        self.gaussian_parameters = nn.ModuleDict(
            {
                'mean':  nn.Linear(hidden_dims[-1] * current_size, self.latent_dim),
                'variance': nn.Linear(hidden_dims[-1] * current_size, self.latent_dim)
            }
        )

        self.decoder = nn.Linear(self.latent_dim, num_classes) # torchvision.ops.MLP(self.latent_dim, [self.latent_dim, self.latent_dim, num_classes])

    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        # Split the result into mu and var components of the latent Gaussian distribution
        mu = self.gaussian_parameters['mean'](result)
        log_var = self.gaussian_parameters['variance'](result) if self.latent_smoothing else 0
        return [mu, log_var]
    
    def reparameterize(self, mu, log_var):
        if not self.latent_smoothing:
            return mu
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        result = self.decoder(z)
        return result

    def forward(self, input):
        
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

        mu_value = 0 # self.mu ** 2
        kld_loss = torch.mean(-0.5 * torch.sum(1 + self.log_var - mu_value - self.log_var.exp(), dim = 1), dim = 0) # Analytic KL Divergence Loss from isotropic gaussian

        classifier_weight_loss = torch.sum(torch.norm(self.decoder.weight, dim=1) ** 2) if float(self.config.lam) > 0 else 0

        # selected_indices = 
        # selected_means = self.mu[:, selected_indices]
        # mean = torch.mean(torch.vstack(list(selected_means)), dim=0)

        labels_stacked = labels.reshape(-1)
        mu_stacked = torch.vstack(list(self.mu))

        margin_loss = 0.0001
        for c in range(self.num_classes):
            indices = (labels_stacked == c).nonzero().flatten()
            selected_mus = mu_stacked[indices]
            mu_mean = torch.mean(selected_mus, dim=0)

            other_indices = (labels_stacked != c).nonzero().flatten()
            others = mu_stacked[other_indices]

            norm_distances = torch.norm(mu_mean - others, dim=1)

            margin = self.config.margin - norm_distances
            margin[margin < 0] = 0
            margin_loss += torch.sum(margin)

        print(label_loss.item(), kld_loss.item(), margin_loss.item())
        print(label_loss.item(), self.kl_weight * kld_loss.item(), self.config.margin_weight * margin_loss.item())

        loss = label_loss + self.kl_weight * kld_loss + self.config.lam * classifier_weight_loss + self.config.margin_weight * margin_loss

        return loss
