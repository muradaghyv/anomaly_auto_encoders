import torch
import torch.nn as nn
torch.cuda.empty_cache()

device = ("cuda" if torch.cuda.is_available() else "cpu")

class AutoEncoder(nn.Module):

    def __init__(self, input_size, hidden_sizes=[64, 32, 16], dropout_rate=0.2):

        super(AutoEncoder, self).__init__()

        # Defining encoder layers
        encoder_layers = []
        current_size = input_size
    
        for hidden_size in hidden_sizes:
            encoder_layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            current_size = hidden_size
        
        self.encoder = nn.Sequential(*encoder_layers)

        # Defining decoder layers
        decoder_layers = []
        hidden_sizes_reversed = hidden_sizes[::-1]
        current_size = hidden_sizes_reversed[0]

        for hidden_size in hidden_sizes_reversed[1:]:
            decoder_layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            current_size = hidden_size
        
        # Final decoder layer for reconstruction
        decoder_layers.append(nn.Linear(current_size, input_size))
        
        self.decoder = nn.Sequential(*decoder_layers)

        # Initializing weight using Xavier initiliazation
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded
    
    def get_reconstruction_error(self, x):

        reconstructed = self.forward(x)
        error = torch.mean((x-reconstructed)**2, dim=1)

        return reconstructed, error
    
    def validate(self, val_loader, criterion, device):

        self.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch[0].to(device)
                reconstructed = self.forward(batch)
                loss = criterion(reconstructed, batch)
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss/num_batches   
        
        return avg_loss