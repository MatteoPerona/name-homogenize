'''
This file contains functions used for visualizing information about the data and exploring results.
'''
import os
import math
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.metrics import precision_recall_fscore_support, precision_score, recall_score, f1_score, accuracy_score

def plot_missingness(df, title):
    """
    Plots a heatmap to visualize missing values in a pandas DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame for which to plot the missingness heatmap.
    title : str
        The title to display above the heatmap, typically indicating the dataset or context.

    Description:
    ------------
    This function creates a heatmap where missing values in the DataFrame are highlighted.
    It uses a `viridis` colormap and suppresses tick labels for rows. This is useful for
    quickly identifying patterns or areas with missing data.

    Output:
    -------
    A heatmap plot is displayed showing the presence of missing values.
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
    plt.title(f'Missingness in {title}')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    plt.show()


def similarity_hist(df, column, title, threshold):
    """
    Plots a histogram of semantic similarity scores and calculates the percentage of values above a threshold.

    Parameters:
    -----------
    df : pandas.DataFrame
        The DataFrame containing the semantic similarity scores.
    column : str
        The name of the column in the DataFrame that contains the similarity scores.
    title : str
        The title for the histogram plot.
    threshold : float
        The threshold value to highlight on the histogram with a vertical dashed line.

    Description:
    ------------
    This function creates a histogram to visualize the distribution of similarity scores
    in the specified column. It overlays a vertical dashed line to indicate the threshold
    and calculates the percentage of values exceeding this threshold. This is useful for
    evaluating the distribution of similarity metrics and assessing how many scores are
    above a critical value.

    Output:
    -------
    A histogram plot is displayed, and the percentage of scores above the threshold is printed.
    """
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
    """
    Computes evaluation metrics for a list of models and their predictions.
    
    Parameters:
    -----------
    dfs : list of pandas.DataFrame
        A list of DataFrames, where each DataFrame contains the true and predicted 
        labels for a specific model.
    models_used : list of str
        A list of model names corresponding to the DataFrames in `dfs`.
    y_true_col : str
        The column name in the DataFrame that contains the true labels.
    y_pred_col : str
        The column name in the DataFrame that contains the predicted labels.
    even_classes : bool, optional (default=True)
        Whether to sample an even distribution of classes for evaluation. If True, 
        it will sample a fixed number of instances per class before computing metrics.
    
    Returns:
    --------
    metrics_df : pandas.DataFrame
        A DataFrame containing the evaluation metrics (precision, recall, F1 score, 
        and accuracy) for each model, sorted by model name.
    
    Metrics Computed:
    -----------------
    - Precision: Proportion of true positives among predicted positives.
    - Recall: Proportion of true positives among actual positives.
    - F1 Score: Harmonic mean of precision and recall.
    - Accuracy: Proportion of correct predictions among all predictions.
    
    Example:
    --------
    metrics = describe_results(dfs, models_used, 'true_labels', 'predicted_labels')
    print(metrics)
    """
    results = []
    
    if even_classes:
        # Look at a sample evenly distributed among classes
        dfs = [result.groupby('classification').head(56) for result in dfs]
    
    for i, df in enumerate(dfs):
        precision = precision_score(df[y_true_col], df[y_pred_col], average='binary')
        recall = recall_score(df[y_true_col], df[y_pred_col], average='binary')
        f1 = f1_score(df[y_true_col], df[y_pred_col], average='binary')
        accuracy = accuracy_score(df[y_true_col], df[y_pred_col])
        
        results.append({
            'model_used': models_used[i],
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        })
    
    metrics_df = pd.DataFrame(results).sort_values(by='model_used')
    return metrics_df


def wilson_confidence_interval(successes, total, confidence=0.95):
    """
    Calculate the Wilson confidence interval for a proportion.

    Parameters:
        successes (int): Number of successes (positive outcomes).
        total (int): Total number of trials.
        confidence (float): Confidence level (default is 0.95 for 95%).

    Returns:
        tuple: Lower and upper bounds of the confidence interval.
    """
    if total == 0:
        raise ValueError("Total number of trials must be greater than 0.")

    # Proportion of successes
    p_hat = successes / total

    # Z-score for the given confidence level
    z = -1 * math.erfc(-confidence / math.sqrt(2)) * math.sqrt(2)

    # Wilson score interval calculations
    denominator = 1 + z**2 / total
    center = p_hat + z**2 / (2 * total)
    margin = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total)

    lower_bound = (center - margin) / denominator
    upper_bound = (center + margin) / denominator

    return lower_bound, upper_bound


def plot_confidence_intervals(csv_paths, output_path, title='', confidence=0.95, data_dir='data/test-results/'):
    """
    Plot scatter points for test results with overlaid confidence intervals.

    Parameters:
        data_dir (str): Directory containing CSV files of test results.
        confidence (float): Confidence level for the intervals (default is 0.95).
    """
    plt.figure(figsize=(10, 6))

    for csv_file in csv_paths:
        # Load data
        file_path = os.path.join(data_dir, csv_file)
        res = pd.read_csv(file_path)

        # Filter data
        positive = res[res['classification'] == True]
        negative = res[res['classification'] == False]

        positive_total = positive.shape[0]
        negative_total = negative.shape[0]

        positive_success = positive[positive['pred'] == True].shape[0]
        negative_success = negative[negative['pred'] == False].shape[0]

        # Calculate probabilities and confidence intervals
        true_positive_p = positive_success / positive_total if positive_total > 0 else 0
        true_negative_p = negative_success / negative_total if negative_total > 0 else 0

        true_positive_ci = wilson_confidence_interval(positive_success, positive_total, confidence)
        true_negative_ci = wilson_confidence_interval(negative_success, negative_total, confidence)

        # Plot data
        scatter = plt.scatter(true_negative_p, true_positive_p, label=os.path.basename(csv_file))
        color = scatter.get_facecolor()[0]  # Extract color of the scatter point

        # Add rectangle for confidence intervals
        rect = Rectangle(
            (true_negative_ci[0], true_positive_ci[0]),  # Bottom-left corner
            true_negative_ci[1] - true_negative_ci[0],  # Width
            true_positive_ci[1] - true_positive_ci[0],  # Height
            alpha=0.2, color=color  # Adjust alpha for opacity
        )
        plt.gca().add_patch(rect)    
    

    # Set legend outside the plot
    plt.legend(
        loc='upper left',  # Legend location
        bbox_to_anchor=(1.05, 1),  # Position outside the plot
        borderaxespad=0,  # Padding between axes and legend
    )
    plt.xlabel('True Negative Probability', fontsize=18)
    plt.ylabel('True Positive Probability', fontsize=18)
    plt.title(f'Confidence Intervals for {title} Test Results', fontsize=20)
    plt.grid(alpha=0.3)
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')

    
    