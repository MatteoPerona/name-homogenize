'''
semantic_similarity.py contains all the similarity logic (e.g. semantic similarity and tf-idf similarity). 
When run from command line this file generates all of the pair similairty data needed to create the eval template. 
'''

import argparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

import numpy as np
import torch
import pandas as pd


# Function to calculate semantic similarity in batches
def calculate_semantic_similarity(names1, names2, model, device, batch_size=32):
    """
    Calculate semantic similarity between two lists of names using a pre-trained SentenceTransformer model.

    Args:
        names1 (list or pd.Series): First list of names to compare.
        names2 (list or pd.Series): Second list of names to compare.
        model (SentenceTransformer): Pre-trained SentenceTransformer model for embedding generation.
        device (torch.device): Device to run the calculations (e.g., CPU or GPU).
        batch_size (int, optional): Number of pairs to process per batch. Defaults to 32.

    Returns:
        np.array: Array of similarity scores between corresponding pairs in names1 and names2.
    """
    # Ensure names1 and names2 are lists
    names1 = names1.tolist() if hasattr(names1, 'tolist') else list(names1)
    names2 = names2.tolist() if hasattr(names2, 'tolist') else list(names2)
    
    similarities = []
    
    # Process in batches to reduce memory usage
    for start_idx in range(0, len(names1), batch_size):
        end_idx = min(start_idx + batch_size, len(names1))
        batch_names1 = names1[start_idx:end_idx]
        batch_names2 = names2[start_idx:end_idx]
        
        # Encode the names
        with torch.no_grad():
            embeddings1 = model.encode(batch_names1, device=device)
            embeddings2 = model.encode(batch_names2, device=device)
        
        # Calculate cosine similarity
        batch_similarities = cosine_similarity(embeddings1, embeddings2).diagonal()
        similarities.extend(batch_similarities)
    
    return np.array(similarities)


# Function to sample and calculate similarity
def process_semantic_similarity(df, column_1, column_2, sample_size, new_col_name='semantic_similarity'):
    """
    Sample rows from a DataFrame and compute semantic similarity between two specified columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        column_1 (str): Name of the first column to compare.
        column_2 (str): Name of the second column to compare.
        sample_size (int): Number of rows to sample for processing.
        new_col_name (str, optional): Name of the new column to store similarity scores. Defaults to 'semantic_similarity'.

    Returns:
        pd.DataFrame: DataFrame with sampled rows and computed semantic similarity scores.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    model = model.to(device)

    # Sample random rows
    sample_df = df.sample(n=sample_size, random_state=42)
    
    # Calculate semantic similarity for the sample
    print(f"Calculating semantic similarity for {sample_size} random rows...")
    sample_df[new_col_name] = calculate_semantic_similarity(
        sample_df[column_1], 
        sample_df[column_2], 
        model, 
        device
    )

    sample_df = sample_df.sort_values(by=new_col_name, ascending=False)
    
    return sample_df


# Calculate semantic similarity of second half of each string pair
def get_second_half(text):
    """
    Get the second half of a string.

    Args:
        text (str): Input string to split.

    Returns:
        str: Second half of the input string.
    """
    words = text.split()
    mid = len(words) // 2
    return ' '.join(words[mid:])

# Function to sample and calculate similarity
def process_second_half_similarity(df, column_1, column_2, sample_size, new_col_name='semantic_similarity'):
    """
    Sample rows and compute semantic similarity using the second half of the text in the specified columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        column_1 (str): Name of the first column to compare.
        column_2 (str): Name of the second column to compare.
        sample_size (int): Number of rows to sample for processing.
        new_col_name (str, optional): Name of the new column to store similarity scores. Defaults to 'semantic_similarity'.

    Returns:
        pd.DataFrame: DataFrame with sampled rows and computed semantic similarity scores based on the second half of the text.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    model = model.to(device)

    # Sample random rows
    sample_df = df.sample(n=sample_size, random_state=42)

    # Get second half
    names_1 = sample_df[column_1].apply(get_second_half)
    names_2 = sample_df[column_2].apply(get_second_half)
    
    # Calculate semantic similarity for the sample
    print(f"Calculating semantic similarity for {sample_size} random rows...")
    sample_df[new_col_name] = calculate_semantic_similarity(
        names_1, 
        names_2, 
        model, 
        device
    )

    sample_df = sample_df.sort_values(by=new_col_name, ascending=False)
    
    return sample_df


# Function to calculate TF-IDF similarity 
def process_tf_idf(df, column_1, column_2, sample_size, new_col_name):
    """
    Sample rows and compute TF-IDF similarity between two specified columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        column_1 (str): Name of the first column to compare.
        column_2 (str): Name of the second column to compare.
        sample_size (int): Number of rows to sample for processing.
        new_col_name (str): Name of the new column to store similarity scores.

    Returns:
        pd.DataFrame: DataFrame with sampled rows and computed TF-IDF similarity scores.
    """
    sample_df = df.sample(n=sample_size, random_state=42)

    # Combine all names into a single list for TF-IDF vectorization
    all_names = list(sample_df[column_1]) + list(sample_df[column_2])

    # Create TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(all_names)

    # Calculate semantic similarity with TF-IDF weighting
    similarities = []
    for i in range(sample_size):
        name1_vector = tfidf_matrix[i]
        name2_vector = tfidf_matrix[i + sample_size]
        similarity = cosine_similarity(name1_vector, name2_vector)[0][0]
        similarities.append(similarity)

    # Add similarities to the sample dataframe
    sample_df[new_col_name] = similarities

    # Sort by similarity in descending order and display results
    results = sample_df.sort_values(new_col_name, ascending=False)

    return results


def create_similarity_data(pairs, sample_size, output_csv_path):
    """
    Generate pairwise similarity data from a given dataset.

    Args:
        pairs (str or pd.DataFrame): Input dataset containing name pairs or path to the CSV file.
        sample_size (int): Number of rows to sample for processing.
        output_csv_path (str): Path to save the output CSV file.

    Returns:
        pd.DataFrame: DataFrame with computed similarity metrics.
    """
    if type(pairs) is str:
        pairs = pd.read_csv(pairs)
    
#     pairs = pairs[['Producer Name_x', 'Abbreviation Name_x', 'Producer Name_y', 'Abbreviation Name_y']]
    pairs = pairs[[
        'Producer Name_x', 
        'Abbreviation Name_x', 
        'Producer Name_y', 
        'Abbreviation Name_y',
        'Region_x',
        'Region_y'
    ]]
    
    # Find similarity between tf-idf vectors representing names
    processed_pairs = process_tf_idf(
        df=pairs, 
        column_1='Producer Name_x', 
        column_2='Producer Name_y', 
        sample_size=sample_size, 
        new_col_name='tf_idf_similarity_name'
    )

    # Find similarity between semantic embeddings of names
    processed_pairs = process_semantic_similarity(
        df=processed_pairs, 
        column_1='Producer Name_x', 
        column_2='Producer Name_y', 
        sample_size=sample_size, 
        new_col_name='semantic_similarity_name'
    )

    # Find similarity between semantic embeddings of second half of names
    processed_pairs = process_second_half_similarity(
        df=processed_pairs, 
        column_1='Producer Name_x', 
        column_2='Producer Name_y', 
        sample_size=sample_size, 
        new_col_name='second_half_similarity_name'
    )

    processed_pairs['second_half_weighted_similarity'] = (processed_pairs['semantic_similarity_name'] + processed_pairs['second_half_similarity_name']) / 2

    processed_pairs.to_csv(output_csv_path, index=False)

    return processed_pairs


def main_internal(sample_size=250_000):
    """
    Internal function to calculate pair similarity and save the results.
    
    Args:
        sample_size (int, optional): Number of samples to process. Defaults to 250,000.
    """
    pairs = 'data/outputs/all_pairs.csv'
    output_csv_path = 'data/outputs/pair_similarity.csv'
    create_similarity_data(
        pairs, 
        sample_size, 
        output_csv_path
    )


def main():
    """
    Command-line interface to calculate pair similarity and save the results.
    """
    parser = argparse.ArgumentParser(description="Create similarity data from input CSV file.")
    # parser.add_argument('pairs', type=str, help="Path to the input CSV file containing pairs of names.")
    parser.add_argument('sample_size', type=int, help="Number of samples to process.")
    # parser.add_argument('output_csv_path', type=str, help="Path to save the output CSV file.")
    
    pairs = 'data/outputs/all_pairs.csv'
    output_csv_path = 'data/outputs/pair_similarity.csv'
    args = parser.parse_args()

    # create_similarity_data(args.pairs, args.sample_size, args.output_csv_path)
    create_similarity_data(pairs, args.sample_size, output_csv_path)


if __name__ == "__main__":
    main()
