#Hourly Impact
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
hourly_impact_png = os.path.join(output_dir, "4_hourly_impact.png")
print("Generating Hourly Impact....\n")

plt.figure(figsize=(10, 6))
sns.boxplot(x='income', y='hours-per-week', data=df, palette='Set2')# hue='occupation' paste this to clear warning in terminal
plt.xlabel('Income Category', fontsize=12)
plt.ylabel('Hours per Week', fontsize=12)
plt.title('Working Hours by Income Level', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(hourly_impact_png, dpi=300, bbox_inches='tight')
plt.close()


