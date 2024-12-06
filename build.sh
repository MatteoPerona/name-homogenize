#!/bin/bash

# Exit script on any error
set -e

# Step 1: Clean the raw data
echo "Cleaning raw data..."
python clean_raw_data.py

# Step 2: Generate pairs using a cartesian product
echo "Generating pairs using a cartesian product..."
python create_pairs.py

# Step 3: Calculate similarity scores
echo "Calculating TF-IDF, semantic, and second-half similarity scores..."
python semantic_similarity.py 250_000 # number of samples 

# Step 4: Create the evaluation template
echo "Creating the evaluation template..."
python create_eval_template.py

# Optional Step: Annotate or use pre-annotated evaluation data
echo "You can now hand annotate the evaluation template or download 'hand_annotated_pairs.csv' before proceeding."

# Step 5: Evaluate data
echo "Generating evaluation data from annotated pairs..."
python evaluate.py

# Completion message
echo "All steps completed successfully."