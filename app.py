import streamlit as st
import pandas as pd
import numpy as np
import statsmodels as sms
import scipy as sp
import seaborn as sns
import sklearn as sl
import joblib
st.set_page_config(page_title= 'Hospital Data Engineering Project')
st.title('Interactive Statistical Analysis Dashboard')

# I used cache_data to prevent Streamlit from rereading the CSV file to improve speed.
@st.cache_data
def load_data():
    clean_df=pd.read_csv('clean_healthcare_dataset.csv')
    return clean_df
clean_df=load_data()
# I'm setting up the loading of the model
@st.cache_resource
def load_fullmodel():
    model= joblib.load('price_model')
    model2= joblib.load('price_model2')
    one_hot=joblib.load('one_hot_encoder')
    ordinal1=joblib.load('ordinal1')
    ordinal2=joblib.load('ordinal2')
    ordinal3=joblib.load('ordinal3')
    ordinal4=joblib.load('ordinal4')
    return model, model2, ordinal1, ordinal2, ordinal3, ordinal4

model, model2, ordina1, ordinal2, ordinal3, ordinal4 = load_fullmodel()


#Setting up the website titles and summart filters

st.subheader('Summary Tables')
st.sidebar.header('Filter and Settings')

num_cols = clean_df.select_dtypes(include=[np.number]).columns.to_list()
select_var= st.sidebar.selectbox('Select variable to display summary tables:',num_cols)

min_value=int(clean_df[select_var].min())
max_value= int(clean_df[select_var].max())
range= st.sidebar.slider('Select value range:',min_value, max_value,(min_value, max_value))

filter_df=clean_df[(clean_df[select_var] >= range[0]) & (clean_df[select_var] <= range[1])]
filter_df
