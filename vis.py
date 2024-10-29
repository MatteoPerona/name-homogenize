import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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