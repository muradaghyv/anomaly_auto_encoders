import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import torch
import torch.nn as nn
import torch.optim as optim
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

joblib.dump(pipeline, "pipeline_updated.pkl") # Saving the pipeline for later usages

# Preparing input data
input_train = torch.tensor(X_train_processed, dtype=torch.float32)
input_val = torch.tensor(X_val_processed, dtype=torch.float32)

input_train = input_train.to(device=device)
input_val = input_val.to(device=device)

# Model definition and training
size = input_train.shape[1]

model = AutoEncoder(size=size) # Model definition
criterion = nn.MSELoss() # Loss definition
optimizer = optim.Adam(params=model.parameters(), lr=0.001) # Optimizer definition
model.to(device=device) # Sending model to device

training_losses = []
validation_losses = []
num_epochs = 250

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(input_train)
    tr_loss = criterion(outputs, input_train)

    tr_loss.backward()
    optimizer.step()

    training_losses.append(tr_loss)

    model.eval()
    with torch.no_grad():
        val_output = model(input_val)
        val_loss = criterion(val_output, input_val)
        validation_losses.append(val_loss)
    
    print(f"Epoch {epoch}: Training loss => {tr_loss}; Validation loss => {val_loss}")

torch.save(model, "anomaly_detection_updated.pth")