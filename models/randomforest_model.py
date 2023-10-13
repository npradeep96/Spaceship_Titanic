"""
Trying to fit an xgboost model
"""

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import GridSearchCV


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

    dataset = pd.read_csv(training_data)
    X_train = dataset.iloc[:, 1:-1]
    Y_train = dataset.Transported

    dataset_validation = pd.read_csv(validation_data)
    passengerIds_validation = dataset_validation.PassengerId
    X_validation = dataset_validation.iloc[:, 1:]

    rf_model = RandomForestClassifier(random_state=42)

    # Performing a grid search of the hyperparameters
    params = {'bootstrap': [True],
              'max_depth': [50, 100, 150],
              'max_features': [2, 4, 6],
              'min_samples_leaf': [2, 4, 6],
              'min_samples_split': [5, 10, 15],
              'n_estimators': [100, 200, 300, 1000]
              }
    search = GridSearchCV(rf_model, param_grid=params, verbose=1, return_train_score=True, n_jobs=1)
    search.fit(X_train, Y_train)
    print('Result of Grid Search ...')
    report_best_scores(search.cv_results_, 3)

    validation_predictions = search.predict(X_validation)
    output_dataframe = pd.DataFrame({'PassengerId': passengerIds_validation,
                                     'Transported': validation_predictions})
    output_dataframe.to_csv('RF_predictions.csv', index=False)
    print('Predictions on validation dataset ....')
    print(validation_predictions)




