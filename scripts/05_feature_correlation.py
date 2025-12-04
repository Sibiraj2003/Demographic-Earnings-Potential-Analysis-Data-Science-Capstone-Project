#Feature Correlation
#Importing neccessary packages and libraries
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
print("Generating Feature correlation plot...")
cor_plt = os.path.join(output_dir, "5_feature_correlation.png")

#splitting numerical features
num_features = df.select_dtypes(include=["int64","float64"]).columns
correlation_mat = df[num_features].corr() #creating correlation matrix for numerical features

plt.figure(figsize=(10,8))
sns.heatmap(correlation_mat, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(cor_plt, dpi=300, bbox_inches='tight')
plt.close()
