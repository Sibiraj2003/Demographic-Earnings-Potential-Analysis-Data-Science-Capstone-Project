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

#Counting Missing Values....
print("\nCounting Missing Values....\n")
missing_file = os.path.join(output_dir, "2_missing_values.csv")
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

#Outlier Detection

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

#Removing Outliers
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

#Plotting starts

#Hours impact
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


#Feature Correlation

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


#Eductional Analysis

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

#Age Analysis

age_analysis_png = os.path.join(output_dir, "7_age_analysis.png")
print("Generating Age Analysis....\n")

plt.figure(figsize=(10,6))
plt.hist(df['age'], bins=30, edgecolor="black", color="purple")
plt.xlabel("Age", fontsize=12)
plt.ylabel("Frequency", fontsize=12)
plt.title('Age Distribution', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(age_analysis_png, dpi=300, bbox_inches='tight')
plt.close()


print("*"*50,"COMPLETED","*"*50)
#END OF CODE
