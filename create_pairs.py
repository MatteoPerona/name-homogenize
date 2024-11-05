'''
create_pairs.py generates the dataset of all possible row pairs between the govenrnment data and the company data. 
'''

import pandas as pd

def cartesian_product(df1, df2, output_csv_path=''):
    """
    Create a dataframe containing all possible combinations of rows from two input dataframes.

    Parameters:
    df1 (DataFrame): The first dataframe.
    df2 (DataFrame): The second dataframe.

    Returns:
    DataFrame: A dataframe containing all possible combinations of rows from df1 and df2.
    """
    # Add a temporary key to both dataframes to enable Cartesian product via merge
    df1['_key'] = 1
    df2['_key'] = 1

    # Perform a merge to create the Cartesian product
    result = pd.merge(df1, df2, on='_key').drop('_key', axis=1)

    # Remove the temporary key column
    df1.drop('_key', axis=1, inplace=True)
    df2.drop('_key', axis=1, inplace=True)

    # Output result csv if path given 
    if output_csv_path != '':
        result.to_csv(output_csv_path, index=False)

    return result

def main():
    producers_1 = pd.read_csv('data/clean/ivorian-cocoa-coop-registry-2017.csv')
    producers_2 = pd.read_csv('data/clean/cocoa-suppliers-compiled-from-importers.csv')

    producers_1 = producers_1[['Producer Name', 'Abbreviation Name', 'Region']]
    producers_2 = producers_2[['Producer Name', 'Abbreviation Name', 'Region']]
    
#     producers_1 = producers_1[['Producer Name', 'Abbreviation Name']]
#     producers_2 = producers_2[['Producer Name', 'Abbreviation Name']]
    
    pairs = cartesian_product(producers_1, producers_2, 'data/outputs/all_pairs.csv')


if __name__ == "__main__":
    main()
