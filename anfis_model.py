# anfis_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ANFIS(nn.Module):
    def __init__(self, n_inputs, n_rules):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_rules = n_rules

        # Parámetros de membresía gaussiana:
        self.mu = nn.Parameter(torch.randn(n_rules, n_inputs))
        self.sigma = nn.Parameter(torch.rand(n_rules, n_inputs))

        # Consecuentes de Takagi-Sugeno (n_inputs parámetros + 1 sesgo por regla):
        self.consequents = nn.Parameter(torch.randn(n_rules, n_inputs + 1))

    def _gaussian_membership(self, x):
        x = x.unsqueeze(1)                    # (batch_size, 1, n_inputs)
        mu = self.mu.unsqueeze(0)             # (1, n_rules, n_inputs)
        sigma = torch.clamp(self.sigma.unsqueeze(0), min=1e-3)
        gauss = torch.exp(-0.5 * ((x - mu) / sigma) ** 2)
        return torch.prod(gauss, dim=2)       # (batch_size, n_rules)

    def forward(self, x):
        batch_size = x.size(0)
        firing = self._gaussian_membership(x)                   # (batch_size, n_rules)
        norm_firing = firing / (firing.sum(dim=1, keepdim=True) + 1e-8)

        x_aug = torch.cat([x, torch.ones(batch_size, 1)], dim=1)  # (batch_size, n_inputs+1)
        rule_out = torch.matmul(x_aug, self.consequents.T)       # (batch_size, n_rules)

        y = torch.sum(norm_firing * rule_out, dim=1, keepdim=True)
        return torch.sigmoid(y)