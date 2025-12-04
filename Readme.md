# 📊 Demographic Earnings Potential Analysis — Data Science Capstone Project

Understanding earning potential across different demographic groups is critical for data-driven workforce planning and policy decisions.  
This project performs an end-to-end **Exploratory Data Analysis (EDA)** on demographic data to uncover key patterns that influence income levels.

---

## 🎯 Project Objectives

The company aims to use demographic analytics to gain competitive advantage.  
This project focuses on:

- Cleaning the dataset by removing missing values  
- Detecting and removing outliers for accurate insights  
- Analyzing the influence of **weekly working hours** on earning potential  
- Identifying features **highly correlated** with earnings  
- Studying the relationship between **years spent to obtain a degree** and income  
- Examining how **age affects earning potential**  

---

## 📂 Dataset Overview

The dataset contains demographic attributes such as:

- Age  
- Education level  
- Years spent in education  
- Weekly working hours  
- Occupation  
- Income category  
- And other socioeconomic variables  

The dataset initially contained missing values, inconsistent entries, and outliers that required preprocessing.

---

## 🧹 Data Cleaning & Preprocessing

Steps performed:

- Removed rows with missing values  
- Detected and eliminated outliers using **IQR** and **Z-Score** techniques  
- Encoded categorical variables (label/one-hot encoding)  
- Validated data distributions and feature consistency  
- Prepared clean dataset for statistical and visual analysis  

---

## 📊 Exploratory Data Analysis (EDA)

### Key Insights Explored:

#### ⭐ Working Hours vs Earnings  
- Higher weekly working hours often correlated with higher earning categories.

#### ⭐ Feature Correlation  
- Identified top features influencing earning potential using **Pearson correlation matrix** & heatmaps.

#### ⭐ Education Duration vs Earnings  
- Studied how years spent obtaining a degree impact income levels.

#### ⭐ Age vs Earnings  
- Analyzed trends showing income progression with age and experience.

### Visuals Used:

- Histograms  
- Scatter plots   
- Boxplots  
- Heatmaps   

---

## 📈 Sample Outputs

<p align="center">
  <img src="output/5_feature_correlation.png" width="45%">
  <img src="output/4_hourly_impact.png" width="45%">
  <img src="output/6_educational_analysis.png" width="45%">
  <img src="output/7_age_analysis.png" width="45%">
</p>

---

## 🧠 Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib  
- Seaborn  

---

## 📦 How to Run the Project

```bash
# Clone the repository
git clone https://github.com/yourusername/ds-demographic-analysis.git

# Navigate into the folder
cd ds-demographic-analysis

# Install dependencies
pip install -r requirements.txt

# Run code
python -m scripts.00_EDA_analysis

📝 Project Structure
css
Copy code
├── data/
│   └── demographic_data.csv
├── notebooks/
│   └── EDA_and_Insights.ipynb
├── src/
│   ├── preprocessing.py
│   ├── outlier_removal.py
│   └── visualization_utils.py
├── plots/
│   ├── age_vs_income.png
│   ├── correlation_heatmap.png
│   └── working_hours_vs_income.png
├── README.md
└── requirements.txt
🚀 Future Improvements
Add regression models to predict earning potential

Build interactive dashboards using Tableau / Power BI / Plotly Dash

Apply clustering (K-Means) for demographic segmentation

Add SHAP-based feature explainability

Automate EDA as a reusable pipeline

🙌 Acknowledgements
This project was completed as part of a Data Science Capstone Program.