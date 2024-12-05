'''
create_eval_template.py generates the template for the evaluation data when run in command line. 
'''

import pandas as pd
import create_pairs
import semantic_similarity

def generate_eval_template(
    processed_pairs, 
    metrics=['second_half_weighted_similarity', 'tf_idf_similarity_name'], 
    output_csv_path='data/outputs/eval_template.csv', 
    n=100):
    """
    Generate an evaluation template for entity resolution tasks.

    This function identifies the top `n` pairs of rows from a DataFrame based on specified 
    metrics, consolidates them into a single DataFrame, and prepares the data for manual 
    classification. The output includes key columns and a default classification column 
    initialized to 0.

    Args:
        processed_pairs (pd.DataFrame): A DataFrame containing pairwise comparisons of entities 
            with associated similarity metrics and other attributes.
        metrics (list of str, optional): List of metric column names to rank the pairs. Defaults 
            to ['second_half_weighted_similarity', 'tf_idf_similarity_name'].
        output_csv_path (str, optional): Path to save the resulting evaluation template as a 
            CSV file. Defaults to 'data/outputs/eval_template.csv'.
        n (int, optional): Number of top-ranked pairs to select per metric. Defaults to 100.

    Returns:
        pd.DataFrame: A DataFrame containing the selected pairs with columns for producer names, 
            abbreviated names, and a default classification column.
    
    Notes:
        - The resulting DataFrame drops duplicates after combining top-ranked pairs from all 
          metrics to ensure unique entries.
        - The output CSV file can be used for manual review and classification tasks.
    """
    
    # Find the top n rows ranked by each metric.
    top_n = []
    for metric in metrics:
        # Get the top n rows ranked by metric
        top_n.append(processed_pairs.sort_values(by=metric, ascending=False).head(n))
        
    # Concatenate the DataFrames and drop duplicate rows
    combined_top_n = pd.concat(top_n)
    
    # Add a column 'classification' and fill it with all 0
    combined_top_n['classification'] = 0
    
    # Select only the producer name and abbreviated name columns
    result = combined_top_n[[
        'Producer Name_x', 
        'Producer Name_y', 
        'Abbreviation Name_x', 
        'Abbreviation Name_y', 
        'classification'
    ]]
    
    result = result.drop_duplicates()
    
    # Write results to csv
    result.to_csv(output_csv_path, index=False)
    
    return result


if __name__ == "__main__":
    '''
    Generate the eval template by pulling the top 200 ranked pairs 
    using second_half_weighted_similarity and tf_idf_similarity_name
    '''
    create_pairs.main()
    semantic_similarity.main_internal()
    
    processed_pairs = pd.read_csv('data/outputs/pair_similarity.csv')
    generate_eval_template(
        processed_pairs=processed_pairs,
        metrics=['second_half_weighted_similarity', 'tf_idf_similarity_name'], 
        output_csv_path='data/outputs/eval_template.csv',
        n=200
    )