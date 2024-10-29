# Name Homogenize

Classifying cocoa producer name pairs as same of different using large language models (LLMs).

We have two csv files to start with. One contains information about cocoa producers contained in a pdf released by the govenrmnet of Ivory Coast. The other contains information about cocoa producers collected from large cocoa importers (e.g. Ferrero, Olam, Nestle, ...). In order to merge the datasets, we need to find all possible row combinations between the CSVs which describe the same entity. <br>

This project aims test various few shot methods to classify which name pairs are describing the same entity starting with just their text, and later using surrounding information. Each method tested is benchmarked against a hand annotated dataset of __ examples.  

