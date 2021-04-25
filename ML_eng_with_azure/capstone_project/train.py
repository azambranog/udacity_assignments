import argparse
import os
from azureml.core.run import Run
from azureml.core import Dataset
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score

run = Run.get_context()
ws = run.experiment.workspace

def retreive_data():
    # get the dataset
    dataset = Dataset.get_by_name(ws, 'hr-data', version='latest')

    # load the Dataset to pandas DataFrame
    data = dataset.to_pandas_dataframe()
    return data


def clean_data(data):
    # drop id column
    # data = data.drop('employee_id', axis=1)
    # drop rows ith NAs
    data = data.dropna()

    # divide features and target
    X = data.drop('is_promoted', axis=1)
    Y = data[['is_promoted']]

    # log transform skew variable
    X['no_of_trainings'] = np.log(X['no_of_trainings'] + 1)

    # minmax scale
    numerical = ['no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 'avg_training_score']
    scaler = MinMaxScaler()
    X[numerical] = scaler.fit_transform(X[numerical])

    # manually convert gender to binary
    X["gender"] = X.gender.apply(lambda s: 1 if s == "f" else 0)

    # onehot encoding for rest of categorical variables
    X = pd.get_dummies(X)

    return X, Y


if __name__ == "__main__":
    # Read command line arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=float, default=50.0, help="Number of estimators of the ensemble")
    parser.add_argument('--learning_rate', type=float, default=1.0, help="Learning rate")

    args = parser.parse_args()

    n_estimators = np.int(args.n_estimators)
    learning_rate = np.float(args.learning_rate)

    run.log("Number of estimators:", n_estimators)
    run.log("Learning Rate:", learning_rate)

    data = retreive_data()
    X, Y = clean_data(data)

    # data split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=155)

    # model training
    model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
    model.fit(X_train, y_train)

    y_predicted = model.predict(X_test)
    score = roc_auc_score(y_test, y_predicted, average="weighted")

    run.log("AUC_weighted", np.float(score))

    os.makedirs('./outputs', exist_ok=True)
    joblib.dump(model, "./outputs/hr-data-adaboost.joblib")