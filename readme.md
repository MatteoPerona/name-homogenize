# Cocoa Co-op Data Homogenization 🌴

This repo evaluates various methods for data homogenization on two datasets of cocoa cooperatives. One from the govenment of Ivory Coast, and the other from cocoa importers. 

## Why does this matter? 

Researchers often face challenges when integrating data that references the same entities but originates from multiple sources, especially when those sources contain minor inconsistencies in how information is presented. Cocoa supply chain data exemplifies this problem, as names and abbreviations of cocoa cooperatives often vary slightly across company and government records, rendering it very difficult to merge datasets. This repo investigates the efficacy of weakly supervised models in classifying pairs of data points to determine if they represent the same cocoa cooperative. 

## Quickstart 🚀

Once you have cloned the repo and created your virtual environment (call it `env`) run:

For Linux/Mac:
```
source env/bin/activate
```
For Windows:
```
env\Scripts\activate
``` 
Install the required dependencies.
```
pip install -r requirements.txt
```
Add `.env` file to your repo.
```
touch .env
```
Add your `OPENAI_API_KEY` to `.env`. Example:
```
OPENAI_API_KEY=...
```
Finally, you can manually install the data to `./data` from [this google drive link](https://drive.google.com/drive/folders/1EdD6FX5vvtcWez4bVDMbCxhvQZdYTAKA?usp=sharing).

`./data` must look like this before you run any code:

```
/data
├── clean
├── eval
│   └── hand_annotated_pairs.csv
├── outputs
├── raw
│   ├── cocoa-suppliers-compiled-from-importers.csv
│   └── ivorian-cocoa-coop-registry-2017.csv
```

These are the minimum files/folders. Everything else can be generated. 

## Generate Data 🔢
If you want to regenerate the evaluation template from raw, run the following scripts in order:

Clean the raw data.
```
python clean_raw_data.py
```
Generate pairs using a cartesian product on the clean data.
```
python create_pairs.py
```
Calculate TF-IDF similarity, semantic similarity, and second-half similarity scores to enhance the evaluation template.
```
python semantic_similarity.py
```
Create the evaluation template.
```
python create_eval_template.py
```
From here you can try to hand annotate the evaluation template or download `hand_annotated_pairs.csv` from the drive folder before generating the evaluaion data using:
```
python evaluate.py
```

## File Descriptions 📂
`clean_raw_data.py`: contains all logic to clean raw data. Run from command line to generate clean data.

`create_pairs.py`: generates the dataset of all possible row pairs between the govenrnment data and the company data. 

`semantic_similarity.py`: contains all the similarity logic (e.g. semantic similarity and tf-idf similarity). When run from command line this file generates all of the pair similairty data needed to create the eval template. 

`create_eval_template.py`: generates the template for the evaluation data when run in command line.

`evaluate.py`: contains logic for evaluating models; outputs results to CSV files when called from command line.

`vis.py`: contains functions used for visualizing information about the data and exploring results.







