"""
Created on Tue 7 May

@author: Clara Driaï-Allègre
"""

#------------------------------------------------------------------------------
#----------------------- Stats for ymin ------------------------
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

#%% Directories

working_directory = '/Users/Clara/Desktop/Multisensory_Project/Tables_N100'
working_directory_All = '/Users/Clara/Desktop/Multisensory_Project/Tables_N100/All'
#%% 

#------------------------------------------------------------------------------
#----------------------- Concatenate all data files ------------------------
#------------------------------------------------------------------------------

#%% Concatenate files (ID, Groups, Conditions)

# List of all subjects
All = ["K01", "K02", "K04", "K05", "K06", "K07", "K08", "K09", "K10", "P01", "P02", "P04", "P05", "P06", "P07", "P08", "P09", "P10"] 
List_K = ["K01", "K02", "K04", "K05", "K06", "K07", "K08", "K09", "K10"]
List_P = ["P01", "P02", "P04", "P05", "P06", "P07", "P08", "P09", "P10"] #"P03"
Cond = "Unpred" # "Unpred" "Pred"

# Function to assign group based on subject ID
def assign_group(subject):
    if subject in List_K:
        return "K"
    elif subject in List_P:
        return "P"
    else:
        return None

#%% Loop over each subject to add the columns: ID, Condition and Group
for subject in All:
    subject_dir = os.path.join(working_directory, subject + "_" + Cond + ".xlsx")
    df = pd.read_excel(subject_dir)
    df["ID"] = subject
    df["Cond"] = Cond
    df["Group"] = df["ID"].apply(assign_group)
    
    rewrite = os.path.join(working_directory, subject + "_New_" + Cond + ".xlsx")
    df.to_excel(rewrite, index=False)

#%% Concatenate for each condition (all participants)

dfs = []

# Loop over each subject to read the Excel file and append it to the list
for subject in All:
    subject_dir = os.path.join(working_directory, subject + "_New_" + Cond + ".xlsx")
    df = pd.read_excel(subject_dir)
    dfs.append(df)

# Concatenate all DataFrames into a single DataFrame
concatenated_df = pd.concat(dfs, ignore_index=True)
df_concat = os.path.join(working_directory, Cond + ".xlsx")
concatenated_df.to_excel(df_concat, index=False)

# Print the concatenated DataFrame
print(concatenated_df)

#%% Concatenate the two created files to get a new one with all data together (participants and conditions)

# Define the paths to the Excel files
file_path1 = os.path.join(working_directory, "Pred.xlsx")
file_path2 = os.path.join(working_directory, "Unpred.xlsx")

# Read the Excel files into DataFrames
df1 = pd.read_excel(file_path1)
df2 = pd.read_excel(file_path2)

# Concatenate the DataFrames into a single DataFrame
concatenated_df = pd.concat([df1, df2], ignore_index=True)
df_all = os.path.join(working_directory, "All.xlsx")
concatenated_df.to_excel(df_all, index=False)

# Print the concatenated DataFrame
print(concatenated_df)


# %%
