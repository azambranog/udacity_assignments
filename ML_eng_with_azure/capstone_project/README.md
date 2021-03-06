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

<img src="img/1_dataset.png" height="280">
 
Once the model was registered, we are able to access it from our workspace by name

<img src="img/2_dataset.png" height="280">

## Automated ML
We first solved the task by using AutoML.

The following configurations were set:
-  Our task is a classification task. 
We used weighted AUC as optimization criteria since it is adequate for imbalanced problems. 
- We set featurization to auto. Meaning autoMl will perform preprocessing such imputation and one hot encoding automatically. 
- To optimize the running time we allowed early stopping if a best score is found. 
The default behavior is to stop the experiment if there is no improvement in the last 10 iterations.
This takes effect only after the 20th iteration.
- We allowed up to 3 parallel runs at the time. 
This could hurt the optimization process, but allows the run to finish faster.
 - Because of the lab limitations we limited the total run of the experiment to  30 minutes.


### Results
We submitted the run and monitor it with the run details widget.

<img src="img/3_automl_submission.png" height="320">

The best model found by AutoML was a voting ensemble with 0.9055 weighted AUC.

<img src="img/4_automl_best_model.png" height="310">

We can check the parameters of the model in the tags. 
The resulting ensemble contains 9 different models with their respective weights.

<img src="img/5_automl_model_params.png" height="360">

The resulting model is already very good. 
We could try increasing the allowed time to run auto ML or the number of runs to look before automatic termination is in place and see if this improves the model.
 


## Hyperparameter Tuning
We created a training script which also cleans the data and fist an Adaboost model.
We chose this algorithm since it generally gives good results, therefore it is worth to try in first step.

We decided to include the number of estimators in the ensemble and the learning rate as parameters to optimize.
We randomly sampled the hyperparameters uniformly.
The number of estimators in the ensemble was kept between 20 and 200, while the learning rate between 0,5 and 2.
The task was to optimize the weighted AUC, so we can compare to AutoML results.

We limited our tuns to 30 since we have limited time for the lab, and allowed 3 runs in parallel. 
In any case, 30 is a good number to start in order to get a feeling is our parameter search space is sensible.

We also activated early stopping based on median value. 
The run will stop if the current model has AUC lower than the median AUC of all previous experiments. 

### Results
We submitted the experiment and monitored it with the run details widget

<img src="img/5_hyper_run_details.png" height="260">

Our best model had a poor performance, the weighted AUC was 0.6324. 
The corresponding parameters were 193 estimators and a learning rate of 1.67.

 <img src="img/6_hyper_best_run.png" height="280">

We notice that the learning rate and the number of estimators for the optimal model are close to our maximum boundary of the sampling space.
This gives an indication that we should expand the sampling grid in the upper direction. 
This could possible lead to better models.

## Model Deployment
We took the resulting model of the AutoML run and proceeded to deploy it as a REST endpoint.
The easiest way to deploy the model is by using the azure web console. 
However, we decided to manually deploy via the python skd.

To manually deploy a model we need the following:
- Registering the model (the pickle file containing the actual model)
- Create an environment to run the model when performing inference. 
AutoMl gives with every run a conda specification file we can use for this.
- Create an entrypoint script used by the web service to process the input and make a prediction.
AutoMl also provides this file for every run.
- Specify where is the model to be deployed and which resources are needed. 
We chose ACI with 1GB memory and 1 cpu core.
- Deploy the webservice using the configurations mentioned above.

 <img src="img/7_service_deployment.png" height="330">

After deployment, we need to wait some minutes until the service is ready for consumption.
After that we can see the model marked as healthy and get the entrypoint URI in the web console.

 <img src="img/7_healty_service.png" height="320">

We can finally make predictions on new data via post request with any client we want. 
In this case we did not specify authentication, but in production systems it is very important to secure the entrypoint with an API key or other authenticacion mechanism.

The following image gives an example of how to make predictions with the API via python

 <img src="img/8_service_query.png" height="280">

In this case th example employee gets a response "1" meaning they will get a promotion. 
The response code 200 indicates all went well with the request.

## Screen Recording
We created a screencast explaining the model deployment process. 
We included a working demo of how to query the model.
The deployed model is available in youtube [https://youtu.be/ELgAsQlBc-0]
