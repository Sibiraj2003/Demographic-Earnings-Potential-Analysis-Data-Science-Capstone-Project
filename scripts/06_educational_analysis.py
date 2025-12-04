#Educational Analysis
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
educational_analysis_png = os.path.join(output_dir, "6_educational_analysis.png")
print("Generating Educational Analysis....\n")

plt.figure(figsize=(10,6))
education_counts = df['education'].value_counts()#getting the count of educational feature
plt.bar(range(len(education_counts)), education_counts.values, color='lightgreen')
plt.xlabel('Education Level', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Education Level Distribution', fontsize=14, fontweight='bold')
plt.xticks(range(len(education_counts)), education_counts.index, rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(educational_analysis_png, dpi=300, bbox_inches='tight')
plt.close()