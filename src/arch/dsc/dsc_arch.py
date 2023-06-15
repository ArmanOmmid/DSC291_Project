
import torch
import torch.nn as nn
import numpy as np

class Smoother(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()

        self.device = device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
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
        
        # z is shape [batch_size, latent_dim]
        z = self.reparameterize(mu, log_var)
        self.z = z
      
        output = self.decode(z)

        return  output
    
    def register_base_criterion(self, criterion):
        self.base_criterion = criterion

    def get_class_averages(self, z, labels, num_classes):
        class_averages = torch.zeros(num_classes, z.size(1), device=self.device)  # Initialize tensor to store class averages
        class_counts = torch.zeros(num_classes, device=self.device)

        for i in range(num_classes):
            class_indices = torch.nonzero(labels == i).squeeze() # Get indices where the label for class i is 1
            if class_indices.numel() == 1:
                class_indices = class_indices.unsqueeze(0)
            class_z = z[class_indices]  # Get the corresponding z values for class i
            class_average = torch.mean(class_z, dim=0)  # Compute the average over z for class i
            class_averages[i] = class_average
            class_counts[i] = len(class_indices)
    
        return class_averages, class_counts
        
    def loss_function(self, output, labels):
        # outputs shape [batch_size, num_classes]
        # labels shape [batch_size]
        
        # --- Fisher Loss ---
        num_classes = output.shape[1]
        z_bar = torch.mean(self.z) # shape [latent_dim]
        z_bar_i, class_counts = self.get_class_averages(self.z, labels, num_classes) # shape [num_classes, latent_dim]
        
        diffs = (z_bar_i - z_bar.unsqueeze(0)) # unsqueeze ensures correct broadcasting dimension
        expanded_diffs = diffs.unsqueeze(-1) # [num_classes, latent_dim, 1]
        
        S_b = torch.sum((expanded_diffs @ expanded_diffs.transpose(1,2)) * class_counts.unsqueeze(-1).unsqueeze(-1),dim=0)
        
        
        S_w = torch.zeros(self.latent_dim, self.latent_dim, device=self.device)
        for z_sample, lbl in zip(self.z, labels):
            diff = z_sample - z_bar_i[lbl]
            expanded_diff = diff.unsqueeze(-1)
            prod = expanded_diff @ expanded_diff.transpose(0,1)
            S_w += prod
            
        print(S_w)
        
        fisher_loss_1 = torch.norm(S_w)
        fisher_loss_2 = torch.norm(S_b)
        
        if not self.latent_smoothing:
            return self.base_criterion(output, labels)

        label_loss = self.base_criterion(output, labels)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + self.log_var - self.mu ** 2 - self.log_var.exp(), dim = 1), dim = 0) # Analytic KL Divergence Loss from isotropic gaussian

        loss = label_loss + self.kl_weight * kld_loss + 0.02*fisher_loss_1 - 0.02*fisher_loss_2

        return loss
