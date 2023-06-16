
import torch
import torch.nn as nn
import torchvision
import numpy as np

class Smoother(nn.Module):
    def __init__(self, num_classes, config):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
        dropout_rate = config.dropout_rate
        for h_dim, pool in zip(hidden_dims, pooling):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size = 3 , stride = 1, padding = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.MaxPool2d((pool,pool)),
                    nn.LeakyReLU(),
                    nn.Dropout(dropout_rate))
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
        log_var = self.gaussian_parameters['variance'](result) if self.latent_smoothing and torch.is_grad_enabled() else 0
        return [mu, log_var]
    
    def reparameterize(self, mu, log_var):
        if not self.latent_smoothing or not torch.is_grad_enabled():
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
            elif class_indices.numel() == 0:
                continue
            class_z = z[class_indices]  # Get the corresponding z values for class i
            class_average = torch.mean(class_z, dim=0)  # Compute the average over z for class i
            class_averages[i] = class_average
            class_counts[i] = len(class_indices)
    
        return class_averages, class_counts
        
    def loss_function(self, output, labels):
        if not self.latent_smoothing or not torch.is_grad_enabled():
            return self.base_criterion(output, labels)

        label_loss = self.base_criterion(output, labels)

        mu_value = 0 # self.mu ** 2
        kld_loss = torch.mean(-0.5 * torch.sum(1 + self.log_var - mu_value - self.log_var.exp(), dim = 1), dim = 0) # Analytic KL Divergence Loss from isotropic gaussian

        classifier_weight_loss = torch.sum(torch.norm(self.decoder.weight, dim=1) ** 2) if float(self.config.classifier_weight_decay) > 0 else 0

        labels_stacked = labels.reshape(-1)
        mu_stacked = torch.vstack(list(self.mu))

        if float(self.config.margin_weight) > 0:
            margin_loss = 0
            for c in range(self.num_classes):

                indices = (labels_stacked == c).nonzero().flatten()
                selected_mus = mu_stacked[indices]

                if selected_mus.shape[0] == 0: continue

                mu_mean = torch.mean(selected_mus, dim=0)
                edge_distance = torch.max(torch.norm(mu_mean - selected_mus, dim=1))

                other_indices = (labels_stacked != c).nonzero().flatten()
                others = mu_stacked[other_indices]
                
                norm_distances = torch.norm(mu_mean - others, dim=1)

                margin = edge_distance + self.config.margin - norm_distances
                margin[margin < 0] = 0
                margin_loss += torch.sum(margin)
        else:
            margin_loss = 0
            
        if float(self.config.fisher_weight_w) > 0 or float(self.config.fisher_weight_b) > 0:
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

            fisher_loss_w = torch.norm(S_w)
            fisher_loss_b = - torch.norm(S_b)
        else:
            fisher_loss = 0

        loss = label_loss + self.kl_weight * kld_loss + self.config.classifier_weight_decay * classifier_weight_loss + self.config.margin_weight * margin_loss + self.config.fisher_weight_w*fisher_loss_w + self.config.fisher_weight_b*fisher_loss_b
        
        #print(self.config.fisher_weight*fisher_loss / loss)

        return loss
    
