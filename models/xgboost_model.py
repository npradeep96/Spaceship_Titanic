"""
Trying to fit an xgboost model
"""

import xgboost as xgb
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def report_best_scores(results, n_top=5):
    """
    Report best scores of models in the grid search
    :param results: An object that is of the form search.cv_results_
    :param n_top: Number of n_top results to write out
    :return: None
    """
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


if __name__ == "__main__":
    """
    Function to build an XGBoost model from the data
    """
    # Read command line arguments that describe file containing input parameters and folder to output simulation results
    parser = argparse.ArgumentParser(description='Input train and validation dataset is a command line argument')
    parser.add_argument('--t', help="Name of training dataset", required=True)
    parser.add_argument('--v', help="Name of validation dataset", required=True)
    args = parser.parse_args()
    training_data = args.t
    validation_data = args.v

    # scaler = StandardScaler(with_mean=True, with_std=False)
    # pca = PCA(n_components=5)

    dataset = pd.read_csv(training_data)
    X_train = dataset.iloc[:, 1:-1]
    Y_train = dataset.Transported
    # X_train = scaler.fit_transform(X_train)
    # X_train = pca.fit_transform(X_train)

    dataset_validation = pd.read_csv(validation_data)
    passengerIds_validation = dataset_validation.PassengerId
    X_validation = dataset_validation.iloc[:, 1:]
    # X_validation = scaler.fit_transform(X_validation)
    # X_validation = pca.transform(X_validation)

    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

    # Impact of learning rate
    params = {"learning_rate": [0.03, 0.1, 0.3]}
    search = GridSearchCV(xgb_model, param_grid=params, verbose=1, return_train_score=True, n_jobs=1)
    search.fit(X_train, Y_train)
    print('Impact of learning rate ...')
    report_best_scores(search.cv_results_, 3)

    # Impact of max_depth
    params = {"max_depth": [2, 4, 6]}
    search = GridSearchCV(xgb_model, param_grid=params, verbose=1, return_train_score=True, n_jobs=1)
    search.fit(X_train, Y_train)
    print('Impact of max tree depth ...')
    report_best_scores(search.cv_results_, 3)

    # Impact of number of estimators used
    params = {"n_estimators": [100, 150, 200]}
    search = GridSearchCV(xgb_model, param_grid=params, verbose=1, return_train_score=True, n_jobs=1)
    search.fit(X_train, Y_train)
    print('Impact of number of estimators used ...')
    report_best_scores(search.cv_results_, 3)

    # Performing a grid search of the hyperparameters
    params = {'objective': ['binary:logistic'],
                  'learning_rate': [0.03, 0.1, 0.3],  # so called `eta` value
                  'max_depth': [6, 10],
                  'subsample': [0.5, 0.7],
                  'colsample_bytree': [0.5, 0.7],
                  'n_estimators': [300, 500, 1000],  # number of trees, change it to 1000 for better results
                  'gamma': [10.0],
                  'alpha': [0.01]
              }
    search = GridSearchCV(xgb_model, param_grid=params, verbose=1, return_train_score=True, n_jobs=1)
    search.fit(X_train, Y_train)
    print('Result of Grid Search ...')
    report_best_scores(search.cv_results_, 3)

    validation_predictions = search.predict(X_validation)
    output_dataframe = pd.DataFrame({'PassengerId': passengerIds_validation,
                                     'Transported': validation_predictions})
    output_dataframe.to_csv('XGBoost_predictions.csv', index=False)
    print('Predictions on validation dataset ....')
    print(validation_predictions)




