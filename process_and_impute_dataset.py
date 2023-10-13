"""
Script to process and impute the datasets
"""
import pandas as pd
import argparse
import feature_processing as fp
import data_imputation as di

if __name__ == "__main__":
    """This script assembles and runs data processing and imputation
    """

    # Read command line arguments that describe file containing input parameters and folder to output simulation results
    parser = argparse.ArgumentParser(description='Input dataset is a command line argument')
    parser.add_argument('--i', help="Name of input dataset", required=True)
    parser.add_argument('--o', help="Name of output file to save the processed dataset", required=True)
    args = parser.parse_args()
    dataset_file_name = args.i
    dataset_cleaned_file_name = args.o

    dataset = pd.read_csv(dataset_file_name)

    # Create new features from PassengerId amd Cabin
    dataset = fp.create_new_features_from_pid(dataset=dataset)
    dataset = fp.create_new_features_from_cabin_numbers(dataset=dataset)
    # dataset = fp.convert_boolean_features_to_float(dataset=dataset)

    # Impute missing values according to different rules
    dataset = di.impute_spending_variables(dataset=dataset)
    dataset = di.impute_cryosleep(dataset=dataset)
    dataset = di.impute_neighbor_for_same_group_number(dataset=dataset)
    dataset = di.impute_to_most_likely_values(dataset=dataset)
    dataset = dataset.drop(['Name', 'Cabin2', 'GroupNumber'], axis=1)

    # dataset = fp.convert_string_to_float(dataset=dataset)
    dataset = fp.categorical_to_onehot(dataset=dataset)

    dataset.to_csv(dataset_cleaned_file_name, index=False)

