import pandas as pd
import numpy as np

import torch 
import torch.nn as nn
import joblib

from sklearn.metrics import accuracy_score, recall_score, f1_score

torch.cuda.empty_cache()

# Defining device
device = ("cuda" if torch.cuda.is_available() else "cpu")

# Uploading the pretrained model and setting eval mode
model = torch.load("best_model.pth")
model = model.to(device)
model.eval()

# Reading the test dataset
df = pd.read_csv("data/test_data.csv", index_col="Unnamed: 0")

df_copy = df.copy()
df_copy["label"] = df_copy["label"].apply(lambda x: 0 if x=="normal." else 1)

# Columns that were used for training
cols = ["protocol_type", "logged_in", "count", "srv_count", "srv_diff_host_rate", "dst_host_count", "dst_host_same_src_port_rate"]
testing_data = df[cols]

# Uploading pipeline that was used for preprocessing training and validation datasets
pipeline = joblib.load("pipeline_updated.pkl")

# Preparing inputs for the evaluation
input_array = pipeline.transform(testing_data)
input_tensor = torch.tensor(input_array, dtype=torch.float32)
input_tensor = input_tensor.to(device=device)

# Feeding the input to the pretrained model
with torch.no_grad():
    outputs = model(input_tensor)

# Loss function definition
criterion = nn.MSELoss()

# Optimal threshold value calculation
optimal_threshold_value = model.optimal_threshold(validation_data=input_tensor, labels=df_copy["label"])
print(f"optimal threshold is: {optimal_threshold_value}")

# Prediction mapping
predictions = []
for i in range(df.shape[0]):
    error = criterion(outputs[i], input_tensor[i])
    if error >= optimal_threshold_value:
        predictions.append(1)
    else:
        predictions.append(0)

# Creating .csv file containing results
results_df = pd.DataFrame({"label": df["label"],
                           "prediction": predictions})

results_df["label"] = results_df["label"].apply(lambda x: 0 if x=="normal." else 1)

# Calculating scores
score = accuracy_score(results_df["label"], results_df["prediction"])
print(f"The accuracy score is {round(score*100, 2)} %.")

recall_0 = recall_score(results_df["label"], results_df["prediction"], pos_label=0)
recall_1 = recall_score(results_df["label"], results_df["prediction"], pos_label=1)

print(f"The Recall score for normal cases: {round(recall_0*100, 2)} %.")
print(f"The Recall score for anomalous cases: {round(recall_1*100, 2)} %.")

# Saving final results_df
results_df.to_csv("results.csv")



