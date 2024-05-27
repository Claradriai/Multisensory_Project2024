"""
Created on Tue 7 May

@author: Clara Driaï-Allègre
"""
#------------------------------------------------------------------------------
#----------------------- Statistical Analysis Second Method------------------------
#------------------------------------------------------------------------------

#%% libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import scipy.stats as stats
from statsmodels.stats.diagnostic import het_white
from scipy.stats import levene
from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import fligner
from scipy.stats import bartlett
from scipy.stats import shapiro
from scipy.stats import levene
from scipy.stats import mannwhitneyu
import scikit_posthocs as sp
from scikit_posthocs import posthoc_dunn

# Parameters to test: 
# ymin (Amplitude)
# ymean (Amplitude)
# xmin (latency)
# ymax (Amplitude) ?

Parameter = "ymin"
#Parameter = "ymean"
# Parameter = "xmin"
# Paramter = ymax (Amplitude)

#%% Import the file with all the data

working_directory = '/Users/Clara/Desktop/Multisensory_Project/Tables_N100'
df_all = os.path.join(working_directory, "All.xlsx")
df = pd.read_excel(df_all)

"""
# Loop for conditions

if Parameter == "ymin":
    test_variable = df["ymin"]
    
elif Parameter == "ymean":
    test_variable = df["ymean"]
    
elif Parameter == "xmin":
    test_variable = df["xmin"]
"""

#%% Remove the outliers for each participant

# Define a function to remove outliers within each group
def identify_outliers(group, column):
    q1 = group[column].quantile(0.25)
    q3 = group[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = (group[column] < lower_bound) | (group[column] > upper_bound)
    return outliers.astype(int)

# Apply the function to each group defined by 'ID'
outlier_column = f"outlier_{Parameter}"
df[outlier_column] = df.groupby('ID').apply(identify_outliers, column=Parameter).reset_index(drop=True)

# Count the number of outliers for each participant
outliers_count = df.groupby('ID')[outlier_column].sum()
# Calculate the total number of outliers
total_outliers = outliers_count.sum()

# Calculate the total number of rows for each participant
total_rows = df.groupby('ID').size()

# Calculate the percentage of outliers for each participant
outliers_percentage = (outliers_count / total_rows) * 100

# Print the results
for participant_id, count, percentage in zip(outliers_count.index, outliers_count, outliers_percentage):
    print(f"Participant {participant_id}: {count} {outlier_column}, {percentage:.2f}%")

# Print the total number of outliers
print(f"Total outliers_{Parameter}: {total_outliers}")

# Remove the outliers
df_clean = df[df[outlier_column] == 0].reset_index(drop=True)

# %% Log-transform the data

def log_transform_with_sign(x):
    return np.sign(x) * np.log(np.abs(x) + 1)

df_clean[f'log_transformed_{Parameter}'] = df_clean[f"{Parameter}"].apply(log_transform_with_sign)

# %% Perform Shapiro-Wilk test for each group combination
for group, group_data in df_clean.groupby(['Group', 'Cond']):
    print(f"Shapiro-Wilk test for {group}:")
    stat, p = shapiro(group_data[f'log_transformed_{Parameter}'])
    print(f"Test Statistic: {stat}, p-value: {p}")
    if p > 0.05:
        print("Data is normally distributed")
    else:
        print("Data is not normally distributed")
    print()

# Plot bell curve for each group combination
for group, group_data in df_clean.groupby(['Group', 'Cond']):
    plt.figure(figsize=(8, 6))
    sns.histplot(group_data[f'log_transformed_{Parameter}'], kde=True, color='blue', stat='density')
    
    # Calculate mean, median, and standard deviation
    mean = group_data[f'log_transformed_{Parameter}'].mean()
    std = group_data[f'log_transformed_{Parameter}'].std()
    median = group_data[f'log_transformed_{Parameter}'].median()
    
    # Add mean, median, and standard deviation to title
    plt.title(f'Bell Curve for Group {group[0]} - Condition {group[1]}\nMean: {mean:.2f}, Median: {median:.2f}, Std: {std:.2f}')
    
    plt.xlabel(f"{Parameter}")
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()

#%% Plot violin plot 

# Plot violin plot for each group/condition
plt.figure(figsize=(10, 6))
sns.violinplot(x='Group', y=f'log_transformed_{Parameter}', hue='Cond', palette = "Set3", data=df_clean, split=True, inner="quart", bw=.2, cut=1, linewidth=1)
plt.title('Violin Plot of ymin for Each Group/Condition')
plt.xlabel('Groups: Controls (K)/Patients (P)')
plt.ylabel('Minimum value of N100 (ymin)')
plt.grid(True)
plt.legend(title='Conditions', loc='upper right')
plt.show()

# %% Performing the two-way Anova (sor far use the non-parametric test as the data is not normally distributed)

# Create model using ols
model = smf.ols(f'log_transformed_{Parameter} ~ C(Cond) + C(Group) + C(Cond):C(Group)', data=df_clean).fit()
# Print summary
print(model.summary())

#%% Post-hoc tests (Tukey HSD)
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# Perform Tukey's HSD test
tukey = pairwise_tukeyhsd(endog=df_clean[f'log_transformed_{Parameter}'], groups=df_clean['Group'] + '-' + df_clean['Cond'], alpha=0.05)
print(tukey)

# %% T-tests

# t-test between groups for each condition
for cond in ['Pred', 'Unpred']:
    group_K = df_clean.loc[(df_clean['Group'] == 'K') & (df_clean['Cond'] == cond), f'log_transformed_{Parameter}']
    group_P = df_clean.loc[(df_clean['Group'] == 'P') & (df_clean['Cond'] == cond), f'log_transformed_{Parameter}']
    
    t_stat, p_val = stats.ttest_ind(group_K, group_P)
    print(f"T-test for {cond} condition between groups K and P:")
    print(f"T-statistic: {t_stat}, P-value: {p_val}")
    if p_val > 0.05:
        print("No significant difference between groups")
    else:
        print("Significant difference between groups")
    print()

# t-test within groups for each condition
for group in ['K', 'P']:
    pred = df_clean.loc[(df_clean['Group'] == group) & (df_clean['Cond'] == 'Pred'), f'log_transformed_{Parameter}']
    unpred = df_clean.loc[(df_clean['Group'] == group) & (df_clean['Cond'] == 'Unpred'), f'log_transformed_{Parameter}']
    
    t_stat, p_val = stats.ttest_rel(pred, unpred)
    print(f"T-test for group {group} between Pred and Unpred conditions:")
    print(f"T-statistic: {t_stat}, P-value: {p_val}")
    if p_val > 0.05:
        print("No significant difference within group")
    else:
        print("Significant difference within group")
    print()