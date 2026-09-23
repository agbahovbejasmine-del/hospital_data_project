# This folder is used for importing my stats functions.
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import f_oneway
import numpy as np
import statsmodels as sms
import scipy as sp
import seaborn as sns
import sklearn as sl
import joblib 
import plotly.express as px


def anova_load():
    anova_data= pd.read_csv('anova_dataset.csv')
    unique= anova_data['Company Group'].unique()
    groups = [anova_data[anova_data['Company Group'] == hospital]['Days Spent'] for hospital in unique]
    print(f"These are the f_statistics and p_values for the ANOVA test for {len(unique)} groups of hospitals with at least 15 patients in the dataset:")
    f_statistic, p_value = f_oneway(*groups)
    result_text= f'''
    These are the f_statistics and p_values for the ANOVA test for {len(unique)} groups of hospitals with at least 15 patients in the dataset:

    F-Statisitic : {f_statistic:.2f} 

    P-Value: {p_value:.2f} 
    '''

    if p_value < 0.05:
        result_text += f'As {p_value:.2f} is less than 0.05, it can be concluded that: \n\n' 
        result_text += "The null hypothesis is rejected: There is a significant difference in the mean days spent among the hospital groups." 
    else:
        result_text += f'As {p_value:.2f} is greater than 0.05, it can be concluded that: \n\n' 
        result_text += "The null hypothesis cannot be rejected: There is no significant difference in the mean days spent among the hospital groups."

    return result_text 

def heat_map():
    clean_df= pd.read_csv('clean_healthcare_dataset.csv')
    contingency_table2 = pd.crosstab(clean_df['Days Spent'],clean_df['Medical Condition'])
    fig= plt.figure(figsize=(10,6))

    sns.heatmap(contingency_table2, annot= True, fmt='d',cmap = 'Greens', linewidths=0.5, cbar= True)
    plt.title(' Contigency Heat Map between Medical Condition and Days Spent',fontsize=15)
    plt.ylabel('Number of Days Spent by Patients', fontsize=10, labelpad=10)
    plt.xlabel(' Number of Patients with a Medical Condition ', fontsize=10, labelpad=10)
    plt.tight_layout()

    return fig