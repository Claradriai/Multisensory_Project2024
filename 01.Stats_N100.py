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

# Parameters: 
# ymin (Amplitude)
# xmin (latency)
# Optional: 
# ymean (Amplitude)
# ymax (Amplitude) 

#Parameter = "ymin"
Parameter = "xmin"

#%% Import the file with all the data

working_directory = '/Users/Clara/Desktop/Multisensory_Project/Tables_N100'
df_all = os.path.join(working_directory, "All.xlsx")
df = pd.read_excel(df_all)

# Convert 'xmin' from seconds to milliseconds
if Parameter == "xmin":
    df[Parameter] = df[Parameter] * 1000

"""
# Loop for conditions

if Parameter == "ymin":
    test_variable = df["ymin"]
    
elif Parameter == "ymean":
    test_variable = df["ymean"]
    
elif Parameter == "xmin":
    test_variable = df["xmin"]
"""

#------------------------------------------------------------------------------
#----------------------- Outlier rejection ------------------------
#------------------------------------------------------------------------------

# Define a function to identify outliers using z-score
def identify_outliers_zscore(group, column, threshold=2.5):
    mean = group[column].mean()
    std = group[column].std()
    z_scores = (group[column] - mean) / std
    outliers = np.abs(z_scores) > threshold
    return outliers.astype(int)
# Apply the function to each group defined by 'ID'
outlier_column = f"outlier_{Parameter}"
df[outlier_column] = df.groupby('ID').apply(identify_outliers_zscore, column=Parameter).reset_index(level=0, drop=True)

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

#------------------------------------------------------------------------------
#----------------------- Assumption check for the ANOVA ------------------------
#------------------------------------------------------------------------------

# %% Perform Shapiro-Wilk test for each group combination
for group, group_data in df_clean.groupby(['Group', 'Cond']):
    print(f"Shapiro-Wilk test for {group}:")
    stat, p = shapiro(group_data[f"{Parameter}"])
    print(f"Test Statistic: {stat}, p-value: {p}")
    if p > 0.05:
        print("Data is normally distributed")
    else:
        print("Data is not normally distributed")
    print()

# Plot bell curve for each group combination
for group, group_data in df_clean.groupby(['Group', 'Cond']):
    plt.figure(figsize=(8, 6))
    sns.histplot(group_data[f"{Parameter}"], kde=True, color='blue', stat='density')
    
    # Calculate mean, median, and standard deviation
    mean = group_data[f"{Parameter}"].mean()
    std = group_data[f"{Parameter}"].std()
    median = group_data[f"{Parameter}"].median()
    
    # Add mean, median, and standard deviation to title
    plt.title(f'Bell Curve for Group {group[0]} - Condition {group[1]}\nMean: {mean:.2f}, Median: {median:.2f}, Std: {std:.2f}')
    
    plt.xlabel(f"{Parameter}")
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()


#%% Perform Levene's test for equality of variances

sample1 = df_clean.loc[(df_clean['Group'] == 'K') & (df_clean['Cond'] == 'Pred'), f"{Parameter}"]
sample2 = df_clean.loc[(df_clean['Group'] == 'K') & (df_clean['Cond'] == 'Unpred'), f"{Parameter}"]
sample3 = df_clean.loc[(df_clean['Group'] == 'P') & (df_clean['Cond'] == 'Pred'), f"{Parameter}"]
sample4 = df_clean.loc[(df_clean['Group'] == 'P') & (df_clean['Cond'] == 'Unpred'), f"{Parameter}"]

statistic, p_value = stats.levene(sample1, sample2, sample3, sample4)

# Print the results
print(f"Levene's Test Results {Parameter}:")
print(f"Test Statistic: {statistic}")
print(f"P-value: {p_value}")
# Interpret the results
if p_value > 0.05:
    print(f"The variances of the four samples are equal ({Parameter})")
else:
    print(f"The variances of the four samples are not equal ({Parameter})")


#%% Plot violin plot for each group/condition (amplitude) 
 
plt.figure(figsize=(10, 6))
ax = sns.violinplot(x='Group', y=f"{Parameter}", hue='Cond', palette="Set3", data=df_clean, split=True, inner="quart", bw=.2, cut=1, linewidth=1)
plt.title('N100 peak amplitude by Group and Condition')
plt.xlabel('Groups')
plt.xticks(ticks=[0, 1], labels=['Controls', 'Patients'])  # Adjust labels as needed
plt.ylabel('N100 amplitude (in uV)')
plt.grid(False)
plt.legend(title='Conditions', loc='upper right')
# Customize the legend
handles, labels = ax.get_legend_handles_labels()
new_labels = ['Predictable', 'Unpredictable']  # Custom labels
plt.legend(handles, new_labels, title='Conditions', loc='upper right')
plt.show()

#%% Plot violin plot for each group/condition (latency) 
 
plt.figure(figsize=(10, 6))
ax = sns.violinplot(x='Group', y=f"{Parameter}", hue='Cond', palette="Set3", data=df_clean, split=True, inner="quart", bw=.2, cut=1, linewidth=1)
plt.title('N100 peak latency by Group and Condition')
plt.xlabel('Groups')
plt.xticks(ticks=[0, 1], labels=['Controls', 'Patients'])  # Adjust labels as needed
plt.ylabel('N100 latency (in ms)')
plt.grid(False)
plt.legend(title='Conditions', loc='upper right')
# Customize the legend
handles, labels = ax.get_legend_handles_labels()
new_labels = ['Predictable', 'Unpredictable']  # Custom labels
plt.legend(handles, new_labels, title='Conditions', loc='upper right')
plt.show()

#------------------------------------------------------------------------------
#------------- Parametric tests (mixedlm/ANOVA + post-hoc + t-tests)-----------
#------------------------------------------------------------------------------

# %% Performing the two-way Anova (sor far use the non-parametric test as the data is not normally distributed)

# Define the mixed-effects model
model_mx = smf.mixedlm(f'{Parameter} ~ C(Group) * C(Cond)', data=df_clean, groups=df_clean['ID'])

# Fit the model
result = model_mx.fit()

# Print the results
print(result.summary())

"""
# Performing the two-way Anova (without the 'ID' predictor)

# Create model using ols
model = smf.ols(f'{Parameter} ~ C(Cond) * C(Group)', data=df_clean).fit()
# Print summary
print(model.summary())

"""

#%% Post-hoc tests (Tukey HSD)
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# Perform Tukey's HSD test
tukey = pairwise_tukeyhsd(endog=df_clean[f'{Parameter}'], groups=df_clean['Group'] + '-' + df_clean['Cond'], alpha=0.05)
print(tukey)

# %% T-tests

# t-test between groups for each condition
for cond in ['Pred', 'Unpred']:
    group_K = df_clean.loc[(df_clean['Group'] == 'K') & (df_clean['Cond'] == cond), f'{Parameter}']
    group_P = df_clean.loc[(df_clean['Group'] == 'P') & (df_clean['Cond'] == cond), f'{Parameter}']
    
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
    pred = df_clean.loc[(df_clean['Group'] == group) & (df_clean['Cond'] == 'Pred'), f'{Parameter}']
    unpred = df_clean.loc[(df_clean['Group'] == group) & (df_clean['Cond'] == 'Unpred'), f'{Parameter}']
    
    t_stat, p_val = stats.ttest_rel(pred, unpred)
    print(f"T-test for group {group} between Pred and Unpred conditions:")
    print(f"T-statistic: {t_stat}, P-value: {p_val}")
    if p_val > 0.05:
        print("No significant difference within group")
    else:
        print("Significant difference within group")
    print()


#------------------------------------------------------------------------------
#--------- Nonparametric tests (Wilcoxon, Mann-Withney, Fridmann) -------------
#------------------------------------------------------------------------------

# %% Split the data into Pred and Unpred for each group
Pred_K = df_clean.loc[(df_clean['Group'] == 'K') & (df_clean['Cond'] == 'Pred'), f"{Parameter}"]
Unpred_K = df_clean.loc[(df_clean['Group'] == 'K') & (df_clean['Cond'] == 'Unpred'), f"{Parameter}"]
Pred_P = df_clean.loc[(df_clean['Group'] == 'P') & (df_clean['Cond'] == 'Pred'), f"{Parameter}"]
Unpred_P = df_clean.loc[(df_clean['Group'] == 'P') & (df_clean['Cond'] == 'Unpred'), f"{Parameter}"]

# %% Performing a Mann-Whitney test (comparison between groups, all conditions)
K_group = df_clean.loc[(df_clean['Group'] == 'K'), f"{Parameter}"]
P_group = df_clean.loc[(df_clean['Group'] == 'P'), f"{Parameter}"]

min_lenght_groups = min(len(K_group), len(P_group))

K_group = K_group.iloc[:min_lenght_groups]
P_group = P_group.iloc[:min_lenght_groups]

# Perform the Wilcoxon signed-rank test for each group
statistic_KP, p_value_KP = stats.wilcoxon(K_group, P_group)

# Print the results
print(f"Diff between groups({Parameter}):")
print(f"Test Statistic: {statistic_KP}")
print(f"P-value: {p_value_KP}")
# Interpret the results
if p_value_KP > 0.05:
    print(f"The two groups are equal ({Parameter})")
else:
    print(f"The two groups are not equal ({Parameter})")

# %% Performing the non-parametric test (Wilcoxon: intra-group comparison)

# Find the minimum length among the two sets of samples
min_length_K = min(len(Pred_K), len(Unpred_K))
min_length_P = min(len(Pred_P), len(Unpred_P))

# Trim the longer samples to match the minimum length using iloc
Pred_K = Pred_K.iloc[:min_length_K]
Unpred_K = Unpred_K.iloc[:min_length_K]
Pred_P = Pred_P.iloc[:min_length_P]
Unpred_P = Unpred_P.iloc[:min_length_P]

# Perform the Wilcoxon signed-rank test for each group
statistic_K, p_value_K = stats.wilcoxon(Pred_K, Unpred_K)
statistic_P, p_value_P = stats.wilcoxon(Pred_P, Unpred_P)

# Print the results
print(f"For Group K - Predictable VS Unpredictable({Parameter}):")
print(f"Test Statistic: {statistic_K}")
print(f"P-value: {p_value_K}")
# Interpret the results
if p_value_K > 0.05:
    print(f"The two conditions are equal ({Parameter})")
else:
    print(f"The two conditions are not equal ({Parameter})")
    
print(f"\nFor Group P - Predictable VS Unpredictable {Parameter}:")
print(f"Test Statistic: {statistic_P}")
print(f"P-value: {p_value_P}")
# Interpret the results
if p_value_P > 0.05:
    print(f"The two conditions are equal ({Parameter})")
else:
    print(f"The two conditions are not equal ({Parameter})")

# %% Performing a Mann-Whitney test (comparison between groups for each condition)

# Find the minimum length among the two sets of samples
min_length_pred= min(len(Pred_K), len(Pred_P))
min_length_unpred = min(len(Unpred_K), len(Unpred_P))

# Trim the longer samples to match the minimum length using iloc
Pred_K = Pred_K.iloc[:min_length_pred]
Unpred_K = Unpred_K.iloc[:min_length_unpred]
Pred_P = Pred_P.iloc[:min_length_pred]
Unpred_P = Unpred_P.iloc[:min_length_unpred]

# Perform the Mann-Whitney U test for each group
statistic_Pred, p_value_Pred = mannwhitneyu(Pred_K, Pred_P) 
statistic_Unpred, p_value_Unpred = mannwhitneyu(Unpred_K, Unpred_P)

# Print the results
print(f"For Predictable condition - K VS P {Parameter}:")
print(f"Test Statistic: {statistic_Pred}")
print(f"P-value: {p_value_Pred}")
# Interpret the results
if p_value_Pred > 0.05:
    print(f"The two groups are equal ({Parameter})")
else:
    print("The two groups are not equal.")
    
# Print the results
print(f"\nFor Unpredictable condition - K VS P {Parameter}:")
print(f"Test Statistic: {statistic_Unpred}")
print(f"P-value: {p_value_Unpred}")
# Interpret the results
if p_value_Unpred > 0.05:
    print(f"The two groups are equal ({Parameter})")
else:
    print(f"The two groups are not equal ({Parameter})")
    
# %% Performing a Friedmann test (Groups x Conditions)

# Find the minimum length among the two sets of samples
min_length_total = min(len(Pred_K), len(Unpred_K), len(Pred_P), len(Unpred_P))

# Trim the longer samples to match the minimum length using iloc
Pred_K = Pred_K.iloc[:min_length_total]
Unpred_K = Unpred_K.iloc[:min_length_total]
Pred_P = Pred_P.iloc[:min_length_total]
Unpred_P = Unpred_P.iloc[:min_length_total]

from scipy.stats import friedmanchisquare
stat_friedman, p_value_friedman = friedmanchisquare(Pred_K, Unpred_K, Pred_P, Unpred_P)
# Print the results
print(f"\nConditions x Groups ({Parameter}):")
print(f"Test Statistic: {stat_friedman}")
print(f"P-value: {p_value_friedman}")
# Interpret the results
if p_value_friedman > 0.05:
    print(f"The four groups are equal ({Parameter})")
else:
    print(f"The four groups are not equal ({Parameter})")
    
#%% Perform post-hoc Dunn's test with Bonferroni correction
posthoc_result = posthoc_dunn([Pred_K, Unpred_K, Pred_P, Unpred_P], p_adjust='bonferroni')
# Print the results], p_adjust='bonferroni')

# Print the post-hoc results
print(f"Post-hoc Dunn's test results ({Parameter}):")
print(posthoc_result)

#------------------------------------------------------------------------------
#---------- Susbequent analysis: N100 amplitude variability -------------------
#------------------------------------------------------------------------------

# %% Perform Levene's test for comparing the variances of each condition between groups

# Perform Levene's test to compare the variances for Pred condition
statistic_levene_pred, p_value_levene_pred = stats.levene(Pred_K, Pred_P)

# Print the results
print(f"\nLevene's test for comparing the variances of Predictable condition between Groups K and P:")
print(f"Test Statistic: {statistic_levene_pred}")
print(f"P-value: {p_value_levene_pred}")
# Interpret the results
if p_value_levene_pred > 0.05:
    print("The variances of the Predictable condition are equal between groups")
else:
    print("The variances of the Predictable condition are not equal between groups")

# Calculate and print the variances for Pred condition
var_Pred_K = Pred_K.var()
var_Pred_P = Pred_P.var()
print(f"\nVariance of Predictable condition for Group K: {var_Pred_K}")
print(f"Variance of Predictable condition for Group P: {var_Pred_P}")

# Determine which group has higher variance for Pred condition
if var_Pred_K > var_Pred_P:
    print("Group K has higher variance for Predictable condition than Group P")
else:
    print("Group P has higher variance for Predictable condition than Group K")

# Perform Levene's test to compare the variances for Unpred condition
statistic_levene_unpred, p_value_levene_unpred = stats.levene(Unpred_K, Unpred_P)

# Print the results
print(f"\nLevene's test for comparing the variances of Unpredictable condition between Groups K and P:")
print(f"Test Statistic: {statistic_levene_unpred}")
print(f"P-value: {p_value_levene_unpred}")
# Interpret the results
if p_value_levene_unpred > 0.05:
    print("The variances of the Unpredictable condition are equal between groups")
else:
    print("The variances of the Unpredictable condition are not equal between groups")

# Calculate and print the variances for Unpred condition
var_Unpred_K = Unpred_K.var()
var_Unpred_P = Unpred_P.var()
print(f"\nVariance of Unpredictable condition for Group K: {var_Unpred_K}")
print(f"Variance of Unpredictable condition for Group P: {var_Unpred_P}")

# Determine which group has higher variance for Unpred condition
if var_Unpred_K > var_Unpred_P:
    print("Group K has higher variance for Unpredictable condition than Group P")
else:
    print("Group P has higher variance for Unpredictable condition than Group K")


# %% Compare variance between P and K groups (combining both conditions)

# Concatenate the data for each group across the two conditions
combined_K = pd.concat([Pred_K, Unpred_K])
combined_P = pd.concat([Pred_P, Unpred_P])

# Perform Levene's test to compare the variances between groups K and P
statistic_levene_combined, p_value_levene_combined = stats.levene(combined_K, combined_P)

# Print the results
print(f"\nLevene's test for comparing the variances between Groups K and P (combining both conditions):")
print(f"Test Statistic: {statistic_levene_combined}")
print(f"P-value: {p_value_levene_combined}")
# Interpret the results
if p_value_levene_combined > 0.05:
    print("The variances are equal between groups K and P (combining both conditions)")
else:
    print("The variances are not equal between groups K and P (combining both conditions)")

# Calculate and print the variances for combined conditions
var_combined_K = combined_K.var()
var_combined_P = combined_P.var()
print(f"\nVariance for Group K (combining both conditions): {var_combined_K}")
print(f"Variance for Group P (combining both conditions): {var_combined_P}")

# Determine which group has higher variance for combined conditions
if var_combined_K > var_combined_P:
    print("Group K has higher variance (combining both conditions) than Group P")
else:
    print("Group P has higher variance (combining both conditions) than Group K")

# %%
# %% Perform Levene's test for comparing the variances of each condition between groups

# Perform Levene's test to compare the variances for Pred condition
statistic_levene_pred, p_value_levene_pred = stats.levene(Pred_K, Unpred_K)

# Print the results
print(f"\nLevene's test for comparing the variances of Predictable condition between Groups K and P:")
print(f"Test Statistic: {statistic_levene_pred}")
print(f"P-value: {p_value_levene_pred}")
# Interpret the results
if p_value_levene_pred > 0.05:
    print("The variances of the condition are equal in the control group")
else:
    print("The variances of the condition are equal in the control group")

# Calculate and print the variances for Pred condition
var_Pred_K = Pred_K.var()
var_Unpred_K = Unpred_K.var()
print(f"\nVariance of Predictable condition for Group K: {var_Pred_K}")
print(f"Variance of Unpredictable condition for Group P: {var_Unpred_K}")


# Perform Levene's test to compare the variances for Unpred condition
statistic_levene_unpred, p_value_levene_unpred = stats.levene(Pred_P, Unpred_P)

# Print the results
print(f"\nLevene's test for comparing the variances of Unpredictable condition between Groups K and P:")
print(f"Test Statistic: {statistic_levene_unpred}")
print(f"P-value: {p_value_levene_unpred}")
# Interpret the results
if p_value_levene_unpred > 0.05:
    print("The variances of the condition are equal in the patient group")
else:
    print("The variances of the condition are equal in the patient group")

# Calculate and print the variances for Unpred condition
var_Pred_P= Pred_P.var()
var_Unpred_P = Unpred_P.var()
print(f"\nVariance of predictable condition for Group P: {var_Pred_P}")
print(f"Variance of Unpredictable condition for Group P: {var_Unpred_P}")

# %%
