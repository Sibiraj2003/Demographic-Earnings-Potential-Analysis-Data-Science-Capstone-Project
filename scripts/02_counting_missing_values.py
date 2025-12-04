#Counting Missing Values
import os 
import numpy as np
import pandas as pd
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, "Adult Dataset", "adult.data")
output_dir = os.path.join(project_root, "output")

# Setting columns 
column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]


missing_file = os.path.join(output_dir, "2_missing_values.csv")
df = pd.read_csv(data_path, names=column_names, skipinitialspace=True)
print("\nCounting Missing Values....\n")
mcounts = {}
for col in  df.columns:
    if df[col].dtypes == 'object': #if value is a character
        count = (df[col]=='?').sum()
    else:#if value is a number
        count = df[col].isnull().sum()
    mcounts[col] = count #adding count to dictionary

missing_df = pd.DataFrame({
    "Column" : mcounts.keys(),
    "Missing Counts" :  mcounts.values(),
    "Percentage" : [(v/len(df)*100) for v in mcounts.values()]
})
print(f"Missing values count {mcounts.values()}")

missing_df = missing_df[missing_df['Missing Counts'] > 0].sort_values('Missing Counts', ascending=False)#sorting the values
missing_df.to_csv(missing_file)