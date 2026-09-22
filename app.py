import streamlit as st
import pandas as pd
import numpy as np
import statsmodels as sms
import scipy as sp
import seaborn as sns
import sklearn as sl
import joblib
import plotly.express as px
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
select_var1= st.sidebar.selectbox('Select  first variable range:',num_cols)
min_value1=int(clean_df[select_var1].min())
max_value1= int(clean_df[select_var1].max())
range1= st.sidebar.slider('Select first value range:',min_value1, max_value1,(min_value1, max_value1))

num_cols2= clean_df.select_dtypes(include=[np.number]).columns.to_list()
select_var2 =st.sidebar.selectbox('Select  second variable range to display summary tables:', num_cols2)
min_value2=int(clean_df[select_var2].min())
max_value2= int(clean_df[select_var2].max())
range2= st.sidebar.slider('Select second value range:',min_value2, max_value2,(min_value2, max_value2)) 

num_cols3= clean_df.select_dtypes(include=[np.number]).columns.to_list()
select_var3 =st.sidebar.selectbox('Select  third variable range to display summary tables:', num_cols3)
min_value3=int(clean_df[select_var3].min())
max_value3= int(clean_df[select_var3].max())
range3= st.sidebar.slider('Select third value range:',min_value3, max_value3,(min_value3, max_value3)) 

# I'm using exceptions to prevent the user from using the same variables twice.

var_list= [select_var1, select_var2, select_var3]
if len(var_list) != len(set(var_list)):
    st.error('You cannot select the same variables for filtering. Please try again.')
    # I use this so that the other error from not defining filter_df is ignored.
    st.stop()

# As I am using multiple 'and' statements, I seperate then by row to make it easier to read since I don't have cells like in Jupyter.
filter_df=clean_df[(clean_df[select_var1] >= range1[0]) & (clean_df[select_var1] <= range1[1])
                   &(clean_df[select_var2] >= range2[0])&(clean_df[select_var2] <= range2[1])
                   &(clean_df[select_var3] >= range3[0])&(clean_df[select_var3] <= range3[1])]


# Creating tabs
tab1, tab2 = st.tabs(['Summary Tables', 'Scatter Graph'])

with tab1:
    st.subheader('Filtered Summary Tables')
    st.dataframe(filter_df, use_container_width=True )

with tab2:
    st.subheader('Scatter Graph Analysis from Summary Tables')
    st.info('This graph shows the data selected from the first two variables.')
    #fig_scat=px.scatter(
    #filter_df,
    #x=select_var1,
    #y=select_var2,
    #title= f'Scatter Graph: {select_var1} against {select_var2}',
    #labels={select_var1 : select_var1, select_var2: select_var2},
    #size='Billing Amount ($)',
    #hover_data= ['Name', 'Medical Condition'],
    #template='plotly_white'
#)
   # st.plotly_chart(fig_scat, use_container_width=True)
  