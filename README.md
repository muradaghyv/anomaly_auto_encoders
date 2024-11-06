05.11.2024 => There are things that need to be done:
1) Data analysis and preparation steps, like encoding and scaling should be done in each training and validation, also test sets so that they are compatible with each other.
2) Model creation and building should be done more accurately and deeper for accurate and better results. It will give us more control over the model. It is better work with the .py files rather than .ipynb files.

06.11.2024 => Things that have done:
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

It can be understood from these results that the model is showing pretty descent results. However, the accuracy for the anomalous cases can be higher.
Therefore, the model structure should be improved for increasing the accuracy for the anomalous cases.

