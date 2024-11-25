# Importing necessary libraries
import numpy as np

from sklearn.metrics import f1_score

import torch
import torch.nn as nn
torch.cuda.empty_cache()

device = ("cuda" if torch.cuda.is_available() else "cpu")

class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_sizes=[64, 32, 16], dropout_rate=0.2):
        """
        Initializes encoder, decoder layers, dropout regularization and Xavier
        initialization of weights.
        Args:
            input_size: the input size of the data fed into the model
            hidden_sizes: the internal layes sizes, there are 2*(hidden_sizes) layers, 
                because auto-encoder layer is symmetrical (encoder and decoder part).
            dropout_rate: dropout rate for regularization
        """
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
        hidden_sizes_reversed = hidden_sizes[::-1] # Decoder structure is symmetrical to the encoder structure
        current_size = hidden_sizes_reversed[0]

        for hidden_size in hidden_sizes_reversed[1:]:
            decoder_layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            current_size = hidden_size
        
        # Final decoder layer for reconstruction the input
        decoder_layers.append(nn.Linear(current_size, input_size))
        
        self.decoder = nn.Sequential(*decoder_layers)

        # Initializing weights using Xavier initiliazation
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """
        Initializes Xavier initialization of weights to all Linear layers.
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded
    
    def get_reconstruction_error(self, x):
        """
        MSE between input data and output of the model.
        High error means anomaly.
        """
        reconstructed = self.forward(x)
        error = torch.mean((x-reconstructed)**2, dim=1)

        return reconstructed, error
    
    def validate(self, val_loader, criterion, device):
        """
        Proper validation step.
        Args:
            val_loader: data loader for the validation data
            criterion: Loss function for the validation
            device: cuda or cpu for calculation
        """
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
    
    def optimal_threshold(self, validation_data, labels):
        """
        Calculates the best threshold value for defining datapoint an anomaly or normal data
        Args:
            validation_data: validation data 
            labels: ground truth values (anomaly or not).
        """
        self.eval()
        with torch.no_grad():
            _, errors = self.get_reconstruction_error(validation_data.to(device))
            
            thresholds = np.percentile(errors.cpu().detach().numpy(), np.arange(0, 100, 5))
            error_arr = errors.cpu().detach().numpy()
            best_f1 = 0
            best_threshold = None

            for threshold in thresholds:
                predictions = (error_arr > threshold)
                f1 = f1_score(predictions, labels)

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

        return best_threshold
    
    def train_model(self, train_loader, val_loader, criterion, optimizer,
              num_epochs, device, early_stopping_patience=10):
        """
        Train function for training the auto-encoder.
        Args:
            train_loader: data loader for training data
            val_loader: data loader for the validation data
            criterion: loss function 
            optimizer: optimizer for backprop
            num_epochs: number of total training epochs
            device: cpu or cuda for calculation
            early_stopping_patience: how many epochs will model wait
                if validation loss doesn't improve
        Outputs:
            training_losses: errors of training
            validation_losses: errors of validation
        """
        training_losses = []
        validation_losses = []
        best_val_loss = float("inf")
        patience_encounter = 0

        for epoch in range(num_epochs):
            self.train()
            epoch_loss = 0.0
            batch_count = 0

            for batch in train_loader:
                batch = batch[0].to(device)

                optimizer.zero_grad()
                reconstructed = self(batch)
                loss = criterion(reconstructed, batch)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1
            
            avg_train_loss = epoch_loss/batch_count
            training_losses.append(avg_train_loss)

            val_loss = self.validate(val_loader, criterion, device)
            validation_losses.append(val_loss)

            if (epoch+1) % 5 == 0:
                print(f"Epoch: {epoch}")
                print(f"Training loss: {avg_train_loss}")
                print(f"Validation loss: {val_loss}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_encounter = 0
                torch.save(self, "best_model.pth")
            else:
                patience_encounter += 1
                if patience_encounter >= early_stopping_patience:
                    print(f"Training stopped after {epoch} epochs!")
                    break
            
        return training_losses, validation_losses