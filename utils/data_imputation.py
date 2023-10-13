"""
Contains functions to perform data imputation
"""
import pandas as pd
import numpy as np


def impute_spending_variables(dataset):
    """
    Function to impute spending variables
    :param dataset: The dataset. Should contain 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck' as features
    :return: dataset: The dataset after processing and adding the new features
    """

    spending_variables = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']

    # First set all the NaNs in the spending variables to 0.0 wherever the passengers are in CryoSleep
    for svar in spending_variables:
        dataset.loc[pd.isna(dataset[svar]) & dataset['CryoSleep'] == 1.0, svar] = 0.0

    # Then impute all the NaN in the spending variables to 0.0 wherever passengers are not in CryoSleep,
    # but their spending in all other categories is 0.0
    for svar in spending_variables:
        for i in range(len(dataset[svar].values)):
            if pd.isna(dataset[svar].iloc[i]):
                sum_of_other_svars = 0.0
                for svar2 in spending_variables:
                    if svar2 != svar and not pd.isna(dataset[svar2].iloc[i]):
                        sum_of_other_svars += dataset[svar2].iloc[i]
                if sum_of_other_svars == 0.0:
                    dataset[svar].iloc[i] = 0.0

    # Then impute all the remaining NaNs to the median values of the column
    for svar in spending_variables:
        dataset.loc[pd.isna(dataset[svar]), svar] = dataset[svar].median()

    for svar in spending_variables:
        print('After this imputation step, the number of NaNs in ' + str(svar) + ' has reduced to '
              + str(len(np.where(pd.isna(dataset[svar]) is True)[0])))

    return dataset


def impute_cryosleep(dataset):
    """
    Impute the NaNs in CryoSleep
    :param dataset: The dataset. Should contain 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck' as features
    :return: dataset: The dataset after processing and adding the new features
    """

    for i in range(len(dataset.CryoSleep.values)):
        # Impute the NaNs in the CryoSleep with False if the sum of values of all the spending variables is non-zero.
        if pd.isna(dataset.CryoSleep.iloc[i]) and dataset['RoomService'].iloc[i] + dataset['FoodCourt'].iloc[i] + \
                dataset['ShoppingMall'].iloc[i] + dataset['Spa'].iloc[i] + dataset['VRDeck'].iloc[i] != 0.0:
            dataset.CryoSleep.iloc[i] = False
        # Impute the NaNs in the CryoSleep with True if the sum of values of all the spending variables is zero.
        # While this is not strictly correct and there may be passengers who are not in sleep but also do not spend,
        # these kind of passengers are small in number.
        elif pd.isna(dataset.CryoSleep.iloc[i]) and dataset['RoomService'].iloc[i] + dataset['FoodCourt'].iloc[i] + \
                dataset['ShoppingMall'].iloc[i] + dataset['Spa'].iloc[i] + dataset['VRDeck'].iloc[i] == 0.0:
            dataset.CryoSleep.iloc[i] = True

    print('After this imputation step, the number of NaNs in CryoSleep has reduced to ' +
          str(len(np.where(pd.isna(dataset['CryoSleep']) is True)[0])))
    return dataset


def impute_neighbor_for_same_group_number(dataset):
    """
    Passengers within the same group number are likely to be from the same home planet, and largely have the same values
    of Cabin1 and Cabin3 values. We can use this rule to impute some data.
    :param dataset: The dataset. Should contain 'HomePlanet', 'GroupNumber', 'Cabin1', 'Cabin3' as features
    :return: dataset: The dataset after processing and adding the new features
    """
    for index in range(len(dataset['HomePlanet'].values)):
        if pd.isna(dataset['HomePlanet'].iloc[index]):
            if index != 0 and dataset['GroupNumber'].iloc[index] == dataset['GroupNumber'].iloc[index-1]:
                dataset['HomePlanet'].iloc[index] = dataset['HomePlanet'].iloc[index-1]
            elif index != len(dataset['GroupNumber']) and dataset['GroupNumber'].iloc[index] \
                    == dataset['GroupNumber'].iloc[index+1]:
                dataset['HomePlanet'].iloc[index] = dataset['HomePlanet'].iloc[index+1]
        if pd.isna(dataset['Cabin1'].iloc[index]):
            if index != 0 and dataset['GroupNumber'].iloc[index] == dataset['GroupNumber'].iloc[index-1]:
                dataset['Cabin1'].iloc[index] = dataset['Cabin1'].iloc[index-1]
            elif index != len(dataset['GroupNumber']) and dataset['GroupNumber'].iloc[index] \
                    == dataset['GroupNumber'].iloc[index+1]:
                dataset['Cabin1'].iloc[index] = dataset['Cabin1'].iloc[index+1]
        if pd.isna(dataset['Cabin3'].iloc[index]):
            if index != 0 and dataset['GroupNumber'].iloc[index] == dataset['GroupNumber'].iloc[index-1]:
                dataset['Cabin3'].iloc[index] = dataset['Cabin3'].iloc[index-1]
            elif index != len(dataset['GroupNumber']) and dataset['GroupNumber'].iloc[index] \
                    == dataset['GroupNumber'].iloc[index+1]:
                dataset['Cabin3'].iloc[index] = dataset['Cabin3'].iloc[index+1]

    print('After this imputation step, the number of NaNs in HomePlanet has reduced to '
          + str(len(np.where(pd.isna(dataset['HomePlanet']) is True)[0])))
    print('After this imputation step, the number of NaNs in Cabin1 has reduced to '
          + str(len(np.where(pd.isna(dataset['Cabin1']) is True)[0])))
    print('After this imputation step, the number of NaNs in Cabin3 has reduced to '
          + str(len(np.where(pd.isna(dataset['Cabin3']) is True)[0])))

    return dataset


def impute_to_most_likely_values(dataset):
    """
    Set the remaining values of HomePlanet, Destination, and VIP status to appropriate values
    :param dataset: The dataset. Should contain 'HomePlanet', 'Destination', 'VIP', 'Age', 'Cabin1', 'Cabin3' features
    :return: dataset: The dataset after processing and adding the new features
    """
    dataset.loc[pd.isna(dataset['HomePlanet']), 'HomePlanet'] = 'Earth'
    # Treat None as a separate feature type
    dataset.loc[pd.isna(dataset['Destination']), 'Destination'] = 'TRAPPIST-1e'
    # Treat None as a separate feature type
    dataset.loc[pd.isna(dataset['VIP']), 'VIP'] = False
    dataset.loc[pd.isna(dataset['Age']), 'Age'] = dataset.Age.median()
    dataset.loc[pd.isna(dataset['Cabin1']), 'Cabin1'] = 'None'
    # Treat None as a separate feature type
    dataset.loc[pd.isna(dataset['Cabin3']), 'Cabin3'] = 'None'
    # Treat None as a separate feature type

    print('After this imputation step, the number of NaNs in HomePlanet has reduced to '
          + str(len(np.where(pd.isna(dataset['HomePlanet']) is True)[0])))
    print('After this imputation step, the number of NaNs in Destination has reduced to '
          + str(len(np.where(pd.isna(dataset['Destination']) is True)[0])))
    print('After this imputation step, the number of NaNs in VIP has reduced to '
          + str(len(np.where(pd.isna(dataset['VIP']) is True)[0])))
    print('After this imputation step, the number of NaNs in Age has reduced to '
          + str(len(np.where(pd.isna(dataset['Age']) is True)[0])))
    print('After this imputation step, the number of NaNs in Cabin1 has reduced to '
          + str(len(np.where(pd.isna(dataset['Cabin1']) is True)[0])))
    print('After this imputation step, the number of NaNs in Cabin3 has reduced to '
          + str(len(np.where(pd.isna(dataset['Cabin3']) is True)[0])))

    return dataset

