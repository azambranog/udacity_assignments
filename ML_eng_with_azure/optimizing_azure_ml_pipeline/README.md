# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
We utilized the UCI bank marketing data set to predict if a customer will suscribe. The data contains contacts made by a Portuguese bank to its clients. Each sample contains personal information about the client (E.g. age, job, housing), information about previous contacts with the client, and financial informaion of the client. A full description of the dataset can be found [here](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing).

We produced a Logistic regression model with SKlearn achieving a 91.279% accuracy. Using AutoML, we found a slightly more accurate model (MaxAbsScaler, LightGBM) with 91.3071% accuracy.

## Scikit-learn Pipeline

For the Scikit-learn pipeline we first created an entrypoint file (train.py). Inside the script, we reard the data into a tabular dataset, perform data cleaning and trained a logistic regression model. We log the accuracy of the model to be used by the hyperparameter tuning  and save the model for future use. The input parameters of the training script are the Regularization Strengt and the Max iterations. These are the parameters to be optimized with Hyperdrive. In the hyperdrive configuration, we indicate out entrypoint script as model and let it find the parameters that optimized the accuracy. 

We selected a random sampler to search in the continous parameter space. Random sampler will randomly select combination of parameters from an specified (unifrom) distribution. It is a good cost effective option to perform a first search of the parameter space. Afterwards one can use the results to better refine the parameter space (more info (here)[https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters]). 

As early termination policy, we chose Bandit policy. This polica allows to early terminate trainign if the optimizytion metric of the current run is isn not within an specified slack factor of the most successful run (more info (here)[https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters#bandit-policy]). It is important to note that in the provided script the metric is only logged once, therefore early termination does not really have any impact.

## AutoML

In the process AutoMl was able to train several different types of models. The best model was of the type MaxAbsScaler LightGBM. The MaxAbsScaler part means that each input feature was scaled by the maximum absolute value in the column. Light Gradient Boosting Machine (LightGBM), this lagorrithm is an optimized boost algorithm that grows trees leaf-wise instead of the conventional level-wise approach.

Note that we explicitelly excluded models using XGBoostClassifier because we were having compatibility issues with the verision of ML suite used.

## Pipeline comparison
- Both models have comparable performance. The Hyperdrive model had 91.279% accuracy while the AutoML model had 91.3071% accuracy.
- For AutoMl we performed data cleaning once beforehand. For the Hyperdrive we had to download the data and clean it on every run of the train script. This however can be easily corrected.
- In Hyperdrive we concentarted only in a single model (logistic regression). Automl was able to try out several types of models.

## Future work

- Use the results of hyperparameter tuning to start a second training round with a finer search space. 
- We sparated the data for automl into train and test split. However we only passed the trin data to AutoML. It is still pending to validate the model in the testdataset.
- Do not perform data import and clean each time we run the train script. We should create a pipeline that perform these steps separately and then runs the Hyperdrive.

