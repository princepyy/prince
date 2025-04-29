import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel, chi2_contingency, shapiro
import matplotlib.ticker as mtick

# Load the data
data = pd.read_excel("sample_employment_data.xlsx")

# Initial info
print(data.head())
print(data.describe())
print("Null values:\n", data.isnull().sum())
print("Duplicate rows:", data.duplicated().sum())

# Objective 1: Workforce Participation Rate Analysis
total_data = data[(data['Distt_Code'] == 0) & 
                  (data['Age_group'] == 'All ages') & 
                  (data['Total_or_Rural_or_Urban'] == 'Total')]

total_data = total_data.copy()
total_data['Workforce_Participation_Rate'] = (total_data['Total_Workers'] / total_data['Total_Persons']) * 100

print("\nWorkforce Participation Rate Summary:")
print(total_data['Workforce_Participation_Rate'].describe())

plt.figure(figsize=(9, 6))
sns.histplot(total_data['Workforce_Participation_Rate'], bins=10, kde=True, color='skyblue')
plt.title('Distribution of Workforce Participation Rate')
plt.xlabel('Participation Rate (%)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Objective 2: Male vs Female Participation Rate
total_data['Male_Participation_Rate'] = (total_data['Total_Workers'] * (total_data['Total_Males'] / total_data['Total_Persons'])) / total_data['Total_Males'] * 100
total_data['Female_Participation_Rate'] = (total_data['Total_Workers'] * (total_data['Total_Females'] / total_data['Total_Persons'])) / total_data['Total_Females'] * 100

# Shapiro normality test
print("\nShapiro Test on Male Participation Rate:")
stat, p = shapiro(total_data['Male_Participation_Rate'])
print(f"Statistic={stat:.4f}, P-Value={p:.4f}")
print("Normal" if p > 0.05 else "Not Normal")

# Paired T-Test
t_stat, p_value = ttest_rel(total_data['Male_Participation_Rate'], total_data['Female_Participation_Rate'])
print(f"\nPaired T-Test P-value: {p_value:.4f}")
print("Significant Difference" if p_value < 0.05 else "No Significant Difference")

# Plotting male vs female
total_data.plot(x='State_Code', y=['Male_Participation_Rate', 'Female_Participation_Rate'], kind='bar', figsize=(14,6))
plt.ylabel('Participation Rate (%)')
plt.title('Male vs Female Workforce Participation by State')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Objective 3: Correlation Among Worker Types
work_type_data = total_data[['Cultivators', 'Agricultural_Labourers', 'Household_Industry_Workers', 'Other_Workers']]
correlation_matrix = work_type_data.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', square=True, linewidths=0.5)
plt.title('Correlation Among Types of Workers')
plt.show()

# Objective 4: Age-wise Workforce Composition
age_data = data[(data['Distt_Code'] == 0) & 
                (data['Total_or_Rural_or_Urban'] == 'Total') & 
                (~data['Age_group'].isin(['All ages']))]

age_grouped = age_data.groupby('Age_group')[['Main_Workers', 'Marginal_Workers']].sum()
age_grouped.plot(kind='bar', stacked=True, figsize=(8,5))
plt.title('Workforce Composition by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Number of Workers')
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(True)
plt.show()

# Objective 5: Urban vs Rural Graduate Level Workforce (Chi-Square Test)
rural = data[(data['Total_or_Rural_or_Urban'] == 'Rural') & (data['Age_group'] == 'All ages')]
urban = data[(data['Total_or_Rural_or_Urban'] == 'Urban') & (data['Age_group'] == 'All ages')]

rural_main = rural['Main_Workers'].sum()
urban_main = urban['Main_Workers'].sum()
rural_total = rural['Total_Workers'].sum()
urban_total = urban['Total_Workers'].sum()

# Chi-square test
chi_table = [[rural_main, rural_total - rural_main],
             [urban_main, urban_total - urban_main]]

chi2, p, dof, expected = chi2_contingency(chi_table)
print("\nChi-Square Test:")
print(f"Chi2 = {chi2:.4f}, P-Value = {p:.4f}")
print("Significant difference" if p < 0.05 else "No significant difference")
