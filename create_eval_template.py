import pandas as pd
import create_pairs
import semantic_similarity

def generate_eval_template(
    processed_pairs, 
    metrics=['second_half_weighted_similarity', 'tf_idf_similarity_name'], 
    output_csv_path='data/outputs/eval_template.csv', 
    n=100):
    
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