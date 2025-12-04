#Outlier Detection 
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
print("OUTLIER DETECTION...")

outlier_file = os.path.join(output_dir, "3_outlier_detection.csv")

outlier_res = [] #Storing outliers result into a list
num_col = df.select_dtypes(include=["int64","float64"]).columns #Finiding columns with numerical values for detection

for col in num_col:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR 
    outliers = df[(df[col]>lower_bound) | (df[col]<upper_bound)] # checking the value is between the lower and upper bound if exists then returns False as a op.
    
    outlier_res.append({
        'Column': col,
        'Q1': Q1,
        'Q3': Q3,
        'IQR': IQR,
        'Lower_Bound': lower_bound,
        'Upper_Bound': upper_bound,
        'Outlier_Count': len(outliers),
        'Outlier_Percentage': (len(outliers)/len(df)*100)
    })
outlier_df = pd.DataFrame(outlier_res)# converting the list to data frame
# print(outlier_df)
outlier_df.to_csv(outlier_file)