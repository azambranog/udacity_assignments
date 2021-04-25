# Prediciton of Employee Promotions with Azure ML 

During this project we demonstrated the main capabilities of Azure ML to train and deploy ML models.
For this purpose we selected a public dataset containing personal information of employees of a company and their performance.

We first let AutoML find a model for the task. 
Then, we manually created  a training script for an AdaBoost model and proceeded to optimize its hypermarameters using an Hyperdrive run.
We compared the result of both experiments and kept the best model.
Finally we deployed the best of the models, making it available to use via a REST API.  

## Dataset

### Overview

https://www.kaggle.com/shivan118/hranalysis
The target is to to predict if an employee is likely o receive a promotion.
*TODO*: Explain about the data you are using and where you got it from.

### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
*TODO* Provide a link to a screen recording of the project in action. Remember that the screencast should demonstrate:
- A working model
- Demo of the deployed  model
- Demo of a sample request sent to the endpoint and its response

## Standout Suggestions
*TODO (Optional):* This is where you can provide information about any standout suggestions that you have attempted.
