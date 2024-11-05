# Cocoa Co-op Data Homogenization 🌴

This repo evaluates various methods for data homogenization on two datasets of cocoa cooperatives. One from the govenment of Ivory Coast, and the other from cocoa importers. 

## Why does this matter? 

Researchers often face challenges when integrating data that references the same entities but originates from multiple sources, especially when those sources contain minor inconsistencies in how information is presented. Cocoa supply chain data exemplifies this problem, as names and abbreviations of cocoa cooperatives often vary slightly across company and government records, rendering it very difficult to merge datasets. This repo investigates the efficacy of weakly supervised models in classifying pairs of data points to determine if they represent the same cocoa cooperative. 

## Quickstart 🚀

Once you have cloned the repo and created your virtual environment. Install the required dependencies. 
```
pip install -r requirements.txt
```
Add a .env file to your repo.
```
touch .env
```
Add your `OPENAI_API_KEY` to the file. For example:
```
OPENAI_API_KEY=...
```
Finally, you can install the data to `./data` from [this google drive link](https://drive.google.com/drive/folders/1EdD6FX5vvtcWez4bVDMbCxhvQZdYTAKA?usp=sharing).

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

These are the minimum files/folders. Everythign else can be generated. 

<!-- ## Data 🔢
## Files 📂 -->







