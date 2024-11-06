import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
torch.cuda.empty_cache()

device = ("cuda" if torch.cuda.is_available() else "cpu")

class AutoEncoder(nn.Module):

    def __init__(self, size):

        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, size)
        )
    
    def forward(self, x):
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded