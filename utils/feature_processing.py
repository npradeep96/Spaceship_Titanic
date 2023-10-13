"""
Contains functions to process features in the training and test data
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def create_new_features_from_pid(dataset):
    """
    Create new features called PassengerNumber and GroupNumber by splitting the '_' in the feature PassengerID
    :param dataset: The dataset. Should contain PassengerID as a feature.
    :return: dataset: The dataset after processing and adding the new features
    """
    group_number = [int(pid.split('_')[0]) for pid in dataset['PassengerId'].values]
    passenger_number = [int(pid.split('_')[1]) for pid in dataset['PassengerId'].values]
    dataset.insert(1, 'GroupNumber', group_number)
    dataset.insert(2, 'PassengerNumber', passenger_number)
    return dataset


def create_new_features_from_cabin_numbers(dataset):
    """
    Create new features from Cabin number, by splitting using '/' as a delimiter
    :param dataset: The dataset. Should contain Cabin as a feature.
    :return: dataset: The dataset after processing and adding the new features
    """
    cabin_numbers = np.array([[None, None, None] if pd.isna(cabin) else cabin.split('/') for cabin in dataset['Cabin'].values])
    dataset.insert(5, 'Cabin1', cabin_numbers[:, 0])
    dataset.insert(6, 'Cabin2', cabin_numbers[:, 1])
    dataset.insert(7, 'Cabin3', cabin_numbers[:, 2])
    return dataset.drop('Cabin', axis=1)


def convert_boolean_features_to_float(dataset):
    """
    Convert boolean and integer features to floating point numbers
    :param dataset: The dataset. Should contain CryoSleep, VIP, and Cabin2 as a feature.
    :return: dataset: The dataset after processing and adding the new features
    """
    dataset['CryoSleep'] = dataset.CryoSleep.astype(float)
    dataset['VIP'] = dataset.VIP.astype(float)
    return dataset


def convert_string_to_float(dataset):
    """
    Ensure that the data columns with categorical values are converted to float
    :param dataset: The dataset. Should contain 'HomePlanet', Destination', 'Cabin1', 'Cabin3' as features
    :return: dataset: The dataset after processing and adding the new features
    """
    home_planets = dataset.HomePlanet.unique()
    for i in range(len(home_planets)):
        dataset.loc[dataset['HomePlanet'] == home_planets[i], 'HomePlanet'] = float(i)
    destinations = dataset.Destination.unique()
    for i in range(len(destinations)):
        dataset.loc[dataset['Destination'] == destinations[i], 'Destination'] = float(i)
    cabin1 = dataset['Cabin1'].unique()
    for i in range(len(cabin1)):
        dataset.loc[dataset['Cabin1'] == cabin1[i], 'Cabin1'] = float(i)
    cabin3 = dataset['Cabin3'].unique()
    for i in range(len(cabin3)):
        dataset.loc[dataset['Cabin3'] == cabin3[i], 'Cabin3'] = float(i)

    dataset['HomePlanet'] = dataset.HomePlanet.astype('float')
    dataset['Destination'] = dataset.Destination.astype('float')
    dataset['Cabin1'] = dataset.Cabin1.astype('float')
    dataset['Cabin3'] = dataset.Cabin3.astype('float')

    return dataset


def categorical_to_onehot(dataset):
    """
        Ensure that the data columns with categorical values are converted to float
        :param dataset: The dataset. Should contain 'HomePlanet', Destination', 'Cabin1', 'Cabin3', 'CryoSleep', 'VIP',
        'PassengerNumber' as features
        :return: dataset: The dataset after processing and adding the new features
        """
    categorical_cols = ['HomePlanet', 'CryoSleep', 'Cabin1', 'Cabin3', 'Destination', 'VIP']
    onehotencoder = OneHotEncoder(drop='first')
    onehotencoder.fit(dataset[categorical_cols])
    transformed_data = onehotencoder.transform(dataset[categorical_cols]).toarray()

    # the above transformed_data is an array so convert it to dataframe
    encoded_data = pd.DataFrame(data=transformed_data, index=dataset.index)
    encoded_data.columns = onehotencoder.get_feature_names(categorical_cols)
    # decoded_data = onehotencoder.inverse_transform(encoded_data)

    # now concatenate the original data and the encoded data using pandas
    concatenated_data = pd.concat([encoded_data, dataset.drop(categorical_cols, axis=1)], axis=1)
    first_column = concatenated_data.pop('PassengerId')
    concatenated_data.insert(0, 'PassengerId', first_column)

    return concatenated_data

def create_new_features_from_pid(dataset):
    """
    Create new features called PassengerNumber and GroupNumber by splitting the '_' in the feature PassengerID
    :param dataset: The dataset. Should contain PassengerID as a feature.
    :return: dataset: The dataset after processing and adding the new features
    """
    group_number = [int(pid.split('_')[0]) for pid in dataset['PassengerId'].values]
    passenger_number = [int(pid.split('_')[1]) for pid in dataset['PassengerId'].values]
    dataset.insert(1, 'GroupNumber', group_number)
    dataset.insert(2, 'PassengerNumber', passenger_number)
    return dataset
