# Project: Operationalize ML with Azure

This project explored the deployment of Ml models and pipelines with Azure. As an exmaple dataset we used the bank marketing dataset (mor information [here](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)). However the focus of this project is not the modeling itself, but its deployment. The following diagram contains a general overview of the project.

<img src="img/archi.png" height="220">

In the first partof the project (upper part of the diagram) we ran an AutoML experiment on the registered dataset. Then we took the resulting best model and deployed it. The result is an enpoint on which we can make predicions on new data points via a POST request.

In the second part of the project we operationalized a Pipline (AutoML). This involves creating the pipeline and making it available via an endpoint. This way, the existing pipeline can be rerun by any authorized used by simply making a POST request to an endpoint.

A quick overview of the project results is avaliable at https://youtu.be/uFgbnzW5HUY (telecast)

## Part 1: Deploy an ML Model

We started from an existing registered dataset containing the bank marketing data
<img src="img/1_registered_dataset.png" height="220">

Then we proceeded to run an AutoML experiment on the dataset
<img src="img/2_auto_ml_experiment.png" height="300">

The best model found by AutoML was a Voting Ensemble with 91.86% accuray. We proceeded to deploy that model.
<img src="img/3_3_best_model.png" height="300">

We also activated applications insights for the deployed model. This is useful to be able to debug and find issues in the deployment.
<img src="img/4_app_insights_enabled.png" height="320">

We can explore the logs to check the most current requests and responses. We used a helper python script to display the logs.
<img src="img/5_logs_py.png" height="420">

Additionally, Azure ML provides documenation of the model. We were able to display the documentation by using a local instance of swagger UI.

<img src="img/6_swagger_methods.png" height="350"><img src="img/7_swagger_responses.png" height="350">

Finally we are able to make predicitons with the published model via POST request. We used a helper python file for this. The image shows the predicted classes for two data points.

<img src="img/8_endpoint_py.png" height="50">

It is important getting a good idea of the performace of the API. For this we used Apache benchmark. It works by sending multiple requests to our deployed model and registering the success rate and the response speed. In the benchmark test all the requests were successful with an average respons etime of 225ms.

<img src="img/9_benchmark.png" height="450">

## Part 2: Deploying a ML Pipeline

We created a pipeline via the Python SKD. We can observe the pipeline runs in the Azure ML portal.
<img src="img/10_pipelines.png" height="150">

We then deployed the piepline so it can be rerun in the future via POST request. We can observe the corresponding endpoint in the pipline endpoints section.
<img src="img/11_pipeline_endpoint.png" height="200">

The pipeline only containes a single AutoML step with the bank dataset as input.
<img src="img/12_dataset_and_automl.png" height="400">

In the overview, we can see the active endpoint. This endpoint can now be used to triger a pipeline run.
<img src="img/13_published_pipeline_overview.png" height="400">

By using the RunDetails widget we can moritor the run of our pipeline
<img src="img/14_pipeline_run_jupyter.png" height="250">

Finaly we used the REST endpoint to trigger a new run of the pipeline. We used a nex experiment "pipeline-res-endpoint" an verified in the portal if the run was created successfully.

<img src="img/15_pipeline_run.png" height="200">

## Outlook and further work.

In this project we were able to deploy a single model as well as a training pipeline. A good next step could be to integrate the published enpoinds with other applications. The model endpoint could be used in a web tool to determine if a specifi customer would accept the marketing offer by the bank. We could also set a trigger to the pipeline entrypoint to run evertime we get a new dataset, this way we would have an upated model without manual effort.
