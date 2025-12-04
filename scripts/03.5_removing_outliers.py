#removing oultliers...
#Importing neccessary packages and libraries
import os 
import numpy as np
import pandas as pd
from datetime import datetime

#setting directories
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, "Adult Dataset", "adult.data")
output_dir = os.path.join(project_root, "output")

column_names = [
    'age', 'workclass', 'fnlwgt', 'education', 'education-num',
    'marital-status', 'occupation', 'relationship', 'race', 'sex',
    'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
]

#using pandas to read the file and store it in a tabular form
print("Loading Dataset...")
df = pd.read_csv(data_path, names=column_names, skipinitialspace=True)

print("*/"*40)
print("Removing Outliers...")
num_col = df.select_dtypes(include=["int64","float64"]).columns #Finiding columns with numerical values for detection
dataset_outlier_free = os.path.join(output_dir, "3.5_cleaned_dataset.csv")
df_clean = df.copy()
for col in num_col:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75) 
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
df_clean.to_csv(dataset_outlier_free)
