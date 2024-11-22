import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
import torch.utils.data
from model import AutoEncoder

import joblib

torch.cuda.empty_cache()

# Defining device
device = ("cuda" if torch.cuda.is_available() else "cpu")

# Reading the dataset
df = pd.read_csv("data.csv")

# Adjusting the label column so that, normal connections are labeled as 0, whereas the others are 1
df["label"] = df["label"].apply(lambda x: 0 if x=="normal." else 1)
normal_data = df[df["label"]==0]

# After EDA, we found out that the necessary columns are as follows
final_cols = ["protocol_type", "logged_in", "count", "srv_count", "srv_diff_host_rate", "dst_host_count", "dst_host_same_src_port_rate"]
final_data = normal_data[final_cols]

# Creating pipeline for preprocessing training, validation and testing datasets 
scaling_cols = ["count", "srv_count", "dst_host_count"] # Numeric columns that need to be scaled
encoding_cols = ["protocol_type"] # Categorical column that needs to be encoded

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, scaling_cols),
        ("cat", categorical_transformer, encoding_cols)
    ],
    remainder="passthrough"
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor)
    ]
)

# Splitting X and y
X = final_data
y = normal_data["label"]

# Splitting data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.75, random_state=42)

# Applying pipeline transforms to the training and validation sets 
X_train_processed = pipeline.fit_transform(X_train)
X_val_processed = pipeline.transform(X_val)

# joblib.dump(pipeline, "pipeline_updated.pkl") # Saving the pipeline for later usages

# Preparing input data
input_train = torch.tensor(X_train_processed, dtype=torch.float32)
input_val = torch.tensor(X_val_processed, dtype=torch.float32)

input_train = input_train.to(device=device)
input_val = input_val.to(device=device)

# Model definition and training
input_size = X_train_processed.shape[1]

# model = AutoEncoder(size=size) # Model definition
# criterion = nn.MSELoss() # Loss definition
# optimizer = optim.Adam(params=model.parameters(), lr=0.001) # Optimizer definition
# model.to(device=device) # Sending model to device

# training_losses = []
# validation_losses = []
# num_epochs = 250

# for epoch in range(num_epochs):
#     model.train()
#     optimizer.zero_grad()

#     outputs = model(input_train)
#     tr_loss = criterion(outputs, input_train)

#     tr_loss.backward()
#     optimizer.step()

#     training_losses.append(tr_loss)

#     model.eval()
#     with torch.no_grad():
#         val_output = model(input_val)
#         val_loss = criterion(val_output, input_val)
#         validation_losses.append(val_loss)
    
#     print(f"Epoch {epoch}: Training loss => {tr_loss}; Validation loss => {val_loss}")

# torch.save(model, "anomaly_detection_updated.pth")


def train_autoencoder(model, train_loader, val_loader, criterion, optimizer,
                      num_epochs, device=device, early_stopping_patience=10):
    
    training_losses = []
    validation_losses = []
    best_val_loss = float("inf")
    patience_encounter = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch in train_loader:
            batch = batch[0].to(device)

            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1
        
        avg_train_loss = epoch_loss/batch_count
        training_losses.append(avg_train_loss)

        val_loss = model.validate(val_loader, criterion, device)
        validation_losses.append(val_loss)

        if (epoch+1) % 5 == 0:
            print(f"Epoch: {epoch}")
            print(f"Training loss: {avg_train_loss}")
            print(f"Validation loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_encounter = 0
            torch.save(model, "best_model.pth")
        else:
            patience_encounter += 1
            if patience_encounter >= early_stopping_patience:
                print(f"Training stopped after {epoch} epochs!")
                break
        
    return training_losses, validation_losses
    
def plot_history(training_losses, validation_losses, save_dir="history.png"):
    if torch.is_tensor(training_losses[0]):
        training_losses = [loss.cpu().detach().numpy() for loss in training_losses]
    if torch.is_tensor(validation_losses[0]):
        validation_losses = [loss.cpu().detach().numpy() for loss in validation_losses]

    # plt.style.use('seaborn')
    sns.set_palette("husl")

    plt.figure(figsize=(12, 6))

    epochs = range(1, len(training_losses)+1)

    plt.plot(epochs, training_losses, label="Training losses", linewidth=2)
    plt.plot(epochs, validation_losses, label="Validation loss", linewidth=2)

    plt.title('Training and Validation Loss Over Time', fontsize=14, pad=15)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    
    # Add grid for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    plt.legend(fontsize=10)
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(save_dir, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print("\nLoss Statistics:")
    print(f"Initial Training Loss: {training_losses[0]:.4f}")
    print(f"Final Training Loss: {training_losses[-1]:.4f}")
    print(f"Best Training Loss: {min(training_losses):.4f}")
    print(f"Initial Validation Loss: {validation_losses[0]:.4f}")
    print(f"Final Validation Loss: {validation_losses[-1]:.4f}")
    print(f"Best Validation Loss: {min(validation_losses):.4f}")
    
    # Calculate and print the improvement
    training_improvement = ((training_losses[0] - training_losses[-1]) / training_losses[0]) * 100
    validation_improvement = ((validation_losses[0] - validation_losses[-1]) / validation_losses[0]) * 100
    
    print(f"\nTraining Loss Improvement: {training_improvement:.2f}%")
    print(f"Validation Loss Improvement: {validation_improvement:.2f}%")


model = AutoEncoder(input_size=input_size) # Model definition
criterion = nn.MSELoss() # Loss definition
optimizer = optim.Adam(params=model.parameters(), lr=0.001) # Optimizer definition
model.to(device=device) # Sending model to device

train_dataset = torch.utils.data.TensorDataset(input_train)
val_dataset = torch.utils.data.TensorDataset(input_val)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = torch.utils.data.DataLoader(
    dataset=val_dataset,
    batch_size=64,
    shuffle=False
)

training_losses, validation_losses = train_autoencoder(model=model, train_loader=train_loader,
                                                       val_loader=val_loader, optimizer=optimizer,
                                                        num_epochs=150, criterion=criterion)

plot_history(training_losses=training_losses, validation_losses=validation_losses)
