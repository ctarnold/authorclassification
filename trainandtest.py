import os
import pandas as pd

directory = "C:/Users/chris/OneDrive/Desktop/Academic/23Fall/JuniorWork/author classification/trainingspreadsheets"
dataframe_all = {}

i = 0

vocabularyset = set()

for filename in os.listdir(directory):
    if filename.endswith(".tsv"): 
        dfwords = pd.read_csv(directory + "/" + filename, usecols = ['surface'], sep = '\t')
        for (columnname, columndata) in dfwords.items():
                for value in columndata.values:
                    vocabularyset.add(value)
                    i = i + 1
        continue
    else:
        continue
print(len(vocabularyset))
print(i)
   