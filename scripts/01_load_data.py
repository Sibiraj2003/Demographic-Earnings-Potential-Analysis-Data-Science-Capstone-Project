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
print(f"Loaded {len(df):,} records with {len(df.columns)} features\n")
print("\nCreating Exploration Summary \n")
#exploration summary creation
data_csv = os.path.join(output_dir, "00_Dataset.csv")
summary_file = os.path.join(output_dir, "1_Exploratory summary.txt")

df.to_csv(data_csv)

with open(summary_file, 'w') as f:
    f.write("*" *80)
    f.write(f"\n EXPLORATION SUMMARY")
    f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}\n")
    f.write("*"*80 + "\n")
    f.write(f"\nDataset Overview\n")
    f.write(f"Records: {len(df):,}")
    f.write(f"\nFeatures: {len(df.columns)}")
    f.write(f"\nDataset Shape: {df.shape}"+"\n")
    f.write("*"*80)
    f.write(f"\nColumn Names ----\n")
    for i, val in enumerate(df.columns, 1):
        f.write(f"{i:2d}. {val}\n")
    f.write("\nFirst 20 rows ---\n")
    f.write("*"*80 + "\n")
    f.write(f"{df.head(20).to_string()}\n")
    f.write("\nData Types ---\n")
    f.write(f"{df.dtypes.to_string()}")
print("\n Creating Statistical Summary...")
# Statistical summary
stats_file = os.path.join(output_dir, "1.5_Statistical Summary.csv")

num_stats = df.describe().T #T is for changing row and columns
# print(num_stats)
num_stats.to_csv(stats_file)
print(f"Stats_file is Saved... {os.path.basename(stats_file)}")

#End of Data Loading and Statistical Analysis


    