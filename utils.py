# This folder is used for importing my stats functions.
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import f_oneway
import numpy as np
import statsmodels as sms
import scipy as sp
import seaborn as sns
import sklearn 
import joblib 
import plotly.express as px
from scipy.stats import norm
from scipy import stats 

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

def normal():
    clean_df= pd.read_csv('clean_healthcare_dataset.csv')
    normo = plt.figure(figsize=(15, 12))

    mu = clean_df['Age'].mean()
    sigma = clean_df['Age'].std()
    plt.hist(clean_df['Age'], bins=30, density=True, alpha=0.5, color='blue', edgecolor='white')
    xmin, xmax = plt.xlim()
    x=np.linspace(xmin, xmax, 100)
    p= norm.pdf(x,mu, sigma)
    plt.plot(x,p,color='darkblue', linewidth=2.5,label=f'Normal curve\n(Mean={mu:.2f},Standard Deviation={sigma:.2f})')
    plt.title('The Normal Curve Distribution of Age',fontsize=14,fontweight='bold')
    plt.xlabel('Age', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(axis='y', linestyle='--',alpha=0.3)
    plt.legend(frameon=True, facecolor='white', edgecolor="none")
    plt.tight_layout()

    return normo

def shapiro():
    clean_df= pd.read_csv('clean_healthcare_dataset.csv')
    data = clean_df['Age'].dropna()
    stat,p_value = stats.shapiro(data)
    test_2 = f'''Shapiro-Wilk Test Statistic: {stat:.4f}

    Shapiro-Wilk Test P-Value: {p_value:.4f}

    Shapiro-Wilk Test P-Value: {p_value:.4f}
    '''
    alpha= 0.05
    if p_value > alpha:
        test_2 += "Fail to reject the null hypothesis: The data is normally distributed."
    else:
        test_2 += "Reject the null hypothesis: The data is not normally distributed."
    return test_2 

def load_fullmodel():
    model= joblib.load('price_model')
    model2= joblib.load('price_model2')
    one_hot=joblib.load('one_hot_encoder')
    ordinal1=joblib.load('ordinal1')
    ordinal2=joblib.load('ordinal2')
    ordinal3=joblib.load('ordinal3')
    ordinal4=joblib.load('ordinal4')

    model_assets ={
        'rf_model':model2,
        'one_hot_enc':one_hot,
        'sort_test' :ordinal1,
        'sort_medical':ordinal2,
        'sort_ins' :ordinal3,
        'sort_admissions':ordinal4
    }
    return model_assets
