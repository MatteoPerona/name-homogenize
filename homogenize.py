import re
import unicodedata
import pandas as pd

def standardize_name(name):
    """
    Standardize a given name by converting it to lowercase, removing accents, and stripping punctuation and extra whitespace.

    Parameters:
    name (str): The name to be standardized.

    Returns:
    str: The standardized name.
    """
    if name is None:
        return None
    name = str(name)
    # Force lowercase
    name = name.lower()
    # Unicode normalization
    name = unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode('ASCII')
    # Remove punctuation and trailing/leading whitespace
    name = re.sub(r'[^\w\s]', '', name).strip()
    name = re.sub(r'\s+', ' ', name)

    return name


def clean_and_output_csv(input_data, output_csv_path):
    """
    Clean the 'Producer Name' and 'Abbreviation Name' columns of a dataframe using the standardize_name function,
    and save the cleaned dataframe to a CSV file.

    Parameters:
    input_data (str or DataFrame): The path to the input CSV file or a DataFrame.
    output_csv_path (str): The path to save the cleaned CSV file.

    Returns:
    DataFrame: The cleaned dataframe.
    """
    # Load the data into a dataframe if input is a CSV path
    if isinstance(input_data, str):
        df = pd.read_csv(input_data)
    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        raise ValueError("input_data must be a file path or a pandas DataFrame")

    # Assert that the required columns exist
    assert 'Producer Name' in df.columns, "The dataframe must contain the 'Producer Name' column."
    assert 'Abbreviation Name' in df.columns, "The dataframe must contain the 'Abbreviation Name' column."

    # Clean the 'Producer Name' and 'Abbreviation Name' columns
    df['Producer Name'] = df['Producer Name'].apply(standardize_name)
    df['Abbreviation Name'] = df['Abbreviation Name'].apply(standardize_name)

    # Drop the 'Country' column if it exists
    if 'Country' in df.columns:
        df = df.drop(['Country'], axis=1)

    # Remove rows where both 'Producer Name' or 'Abbreviation Name' are None
    df.replace('nan', None, inplace=True)
    df = df.dropna(subset=['Producer Name', 'Abbreviation Name'], how='any')
#     df = df.dropna(subset=['Producer Name'])

    # Output the cleaned dataframe to a CSV file
    df.to_csv(output_csv_path, index=False)

    return df


def find_exact_matches_csv(producers_1, producers_2, output_csv_path):
    """
    Find exact matches between the 'Producer Name' columns of two dataframes after standardizing the names, 
    and save the results to a CSV file.

    Parameters:
    producers_1 (DataFrame): The first dataframe containing producer names.
    producers_2 (DataFrame): The second dataframe containing producer names.
    output_csv_path (str): The path to save the CSV file with the exact matches.

    Returns:
    DataFrame: A dataframe containing the exact matches from both input dataframes.
    """
    # Drop null values from both series and keep only unique values
    names_1 = producers_1['Producer Name'].dropna().drop_duplicates(keep='first')
    names_2 = producers_2['Producer Name'].dropna().drop_duplicates(keep='first')

    # Apply standardization to names_1 and names_2
    standardized_names_1 = names_1.apply(standardize_name)
    standardized_names_2 = names_2.apply(standardize_name)

    # Create mappings between original and standardized names
    mapping_1 = dict(zip(standardized_names_1, names_1.index))
    mapping_2 = dict(zip(standardized_names_2, names_2.index))

    # Find the exact matches between standardized names_1 and standardized names_2
    matching_names = standardized_names_1[standardized_names_1.isin(standardized_names_2)]

    # Get the original indexes of matching names from both dataframes
    matches_producers_1 = producers_1.loc[matching_names.map(mapping_1)]
    matches_producers_2 = producers_2.loc[matching_names.map(mapping_2)]

    # Concatenate the matching rows from both dataframes
    exact_matches_df = pd.concat([matches_producers_1.reset_index(drop=True), matches_producers_2.reset_index(drop=True)], axis=1)

    # Output the exact matches to a CSV file
    exact_matches_df.to_csv(output_csv_path, index=False)

    return exact_matches_df


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



