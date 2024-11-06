# Auto-Encoder based Anomaly Detection
This project implements an anomaly detection system using autoencoders, a type of neural network model, on a dataset with labeled "normal" and "anomalous" samples. Autoencoders are commonly used for anomaly detection due to their ability to learn a compressed representation of normal data patterns. This README will guide you through the methodology, installation, usage, and the workflow of this project.

## Table of Contents
- Introduction
- Why use Auto-Encoders for Anomaly Detection?
- Methodoly
- Results
- Future Improvements

## Introduction
Anomaly detection is crucial for identifying unexpected, rare events in data, which can indicate fraud, defects, or other abnormalities. Traditional supervised models rely on labeled data; however, in real-world applications, it is often difficult to obtain enough representative samples of anomalous cases. This is where unsupervised approaches like autoencoders come into play.

This project uses an autoencoder to identify anomalies in data by learning only from the normal data. The autoencoder model is trained to reconstruct normal samples, learning the essential patterns of normal behavior. During testing, if a data point cannot be well-reconstructed, it is classified as an anomaly.

## Why use Auto-Encoders for Anomaly Detection?
Autoencoders are neural networks designed to learn a compressed representation (encoding) of the input data, which they later attempt to reconstruct. For anomaly detection:
- Training Focus: We train the autoencoder only on "normal" data, so the model becomes familiar with the patterns of normality.
- Anomaly Identification: When we feed unseen data, if the autoencoder fails to reconstruct a data point well, this reconstruction error signals that the data point could be anomalous.
- Efficiency: This method is efficient for detecting outliers, especially when anomalous data is rare.

## Methodology
1. Data Preprocessing
    - Scaling and Encoding: Numerical data is scaled, and categorical data is encoded. This project uses a preprocessing pipeline to ensure consistent transformations across training, validation, and testing data.
    - Data Split: The dataset is split into training, validation, and test sets to assess model performance.
2. Building the Autoencoder Model
    - The autoencoder is a neural network with three main parts:
    - Encoder: Compresses the input data into a lower-dimensional representation.
    - Latent Space: The bottleneck layer that holds the compressed knowledge of the input.
    - Decoder: Reconstructs the original data from the compressed representation.
3. Model Training
    - The model is trained on normal data using reconstruction loss (e.g., Mean Squared Error).
    - Early Stopping is used to avoid overfitting, halting training when the validation loss stops decreasing.
4. Anomaly Detection
    - Reconstruction Error Calculation: For each test sample, calculate the reconstruction error.
    - Thresholding: Set a threshold error to distinguish normal samples from anomalies. If the reconstruction error for a sample is above this threshold, it is flagged as anomalous.
5. Evaluation
    - Evaluate the model using metrics like Precision, Recall, and accuracy-score to understand its effectiveness in anomaly detection.

## Results
1) 05.11.2024 => There are things that need to be done:
    1) Data analysis and preparation steps, like encoding and scaling should be done in each training and validation, also test sets so that they are compatible with each other.
    2) Model creation and building should be done more accurately and deeper for accurate and better results. It will give us more control over the model. It is better work with the .py files rather than .ipynb files.

2) 06.11.2024 => Things that have done:
    1) Data preparation steps are updated: duplicates are removed and indexes are resetted for both datasets: training data and test data.
    2) Model building is done separately for more professional view. 
    3) Pipeline is created for scaling the numerical features and encoding the categorical features. This pipeline is used in preprocessing training and validation steps. Then it is used for the test set.

    Things that need to be done:
    1) Model structure needs to be improved, because the final accuracy is approximately 60%. Learning rate scheduling, learning rate adjustments can be done. Validation step can be added to the model building steps.
    2) Overfitting should be checked, if it occurs.

    I have improved the model's performances and metrics are as follows:
    1) The accuracy score is 89.65 %.
    2) The Recall score for normal cases: 95.75 %.
    3) The Recall score for anomalous cases: 79.7 %.

## Future Improvements.
It can be understood from these results that the model is showing pretty descent results. However, the accuracy for the anomalous cases can be higher.
Therefore, the model structure should be improved for increasing the accuracy for the anomalous cases.

