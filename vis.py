'''
This file contains functions used for visualizing information about the data and exploring results.
'''

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score


# Define function to plot missingness heatmap
def plot_missingness(df, title):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title(f'Missingness in {title}')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.show()


def similarity_hist(df, column, title, threshold, ):
    # Create a histogram of semantic similarity scores
    plt.figure(figsize=(8, 4))
    plt.hist(df[column], bins=50, edgecolor='black')
    plt.title(title)
    plt.xlabel('Semantic Similarity')
    plt.ylabel('Frequency')
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Calculate percentage of pairs above threshold
    above_threshold = (df[column] > threshold).mean() * 100
    print(f"Percentage of pairs with similarity > {threshold}: {above_threshold:.2f}%")
    
    
def describe_results(dfs, models_used, y_true_col, y_pred_col, even_classes=True):
    results = []
    
    if even_classes:
        # Look at a sample evenly distributed among classes
        dfs = [result.groupby('classification').head(56) for result in dfs]
    
    for i, df in enumerate(dfs):
        precision = precision_score(df[y_true_col], df[y_pred_col], average='binary')
        recall = recall_score(df[y_true_col], df[y_pred_col], average='binary')
        f1 = f1_score(df[y_true_col], df[y_pred_col], average='binary')
        
        results.append({
            'model_used': models_used[i],
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    metrics_df = pd.DataFrame(results)
    return metrics_df