# Prediciton of Employee Promotions with Azure ML 

During this project we demonstrated the main capabilities of Azure ML to train and deploy ML models.
For this purpose we selected a public dataset containing personal information of employees of a company and their performance.

We first let AutoML find a model for the task. 
Then, we manually created  a training script for an AdaBoost model and proceeded to optimize its hypermarameters using an Hyperdrive run.
We compared the result of both experiments and kept the best model.
Finally we deployed the best of the models, making it available to use via a REST API.  

## Dataset

### Overview
For this project we used a dataset containing information about the employees of a company.
The data includes personal information and work performance metrics for each employee.

The dataset is publicly available for Kaggle users [here](https://www.kaggle.com/shivan118/hranalysis).
Credit: Kaggle user [shivan118](https://www.kaggle.com/shivan118)
 

### Task
The target is to to predict if an employee is likely o receive a promotion given their personal data and work performance metrics.
The dataset is imbalanced with only 5% of the employees in the dataset having obtained a promotion.
The problem is a classical imbalanced binary classification problem.

### Access
To access the dataset we manually registered it with the name "hr-data" as a tabular dataset. 
We took the chance to directly remove columns that are unnecessary for training (e.g. Id). 

<img src="img/1_dataset.png" height="220">
 
Once the model was registered, we are able to access it from our workspace by name

<img src="img/2_dataset.png" height="220">

## Automated ML
We first solved the task by using AutoML. 
Our task is a classification task. 
To optimize the running time we allowed AutoMl to stop early if a best score is found.
We also allowed up to 3 parallel runs and limited the total run time to 30 minutes. 
Finally we chose the weighted AUC as metric since it is an appropriate metric for the imbalanced task.


### Results
We submitted the run and monitor it with the run details widget.

<img src="img/3_automl_submission.png" height="220">

The best model found by AutoML was a voting ensamble with 0.90543 weighted AUC.

<img src="img/4_automl_best_model.png" height="220">


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
