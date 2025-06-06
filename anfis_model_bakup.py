import torch
import torch.nn as nn

class ANFIS(nn.Module):
    def __init__(self, n_inputs, n_rules):
        super(ANFIS, self).__init__()
        self.n_inputs = n_inputs
        self.n_rules = n_rules
        self.pesos = nn.Parameter(torch.randn(n_rules, n_inputs))  
        self.membresia = nn.Parameter(torch.randn(n_rules, 1))  

    def forward(self, x):
        reglas = torch.matmul(x, self.pesos.T)  
        reglas_activadas = torch.softmax(reglas, dim=1)  
        membresia_activada = torch.matmul(reglas_activadas, self.membresia)  
        salida = torch.sigmoid(membresia_activada)  # Forzar para render que el resultado esté entre 0 y 1
        return salida
