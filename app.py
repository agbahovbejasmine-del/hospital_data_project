import streamlit as st
import pandas as pd
import numpy as np
import statsmodels as sms
import scipy as sp
import seaborn as sns
import joblib
import plotly.express as px
from utils import anova_load, heat_map , normal, shapiro, load_fullmodel
from scipy.stats import norm
from scipy import stats
st.set_page_config(page_title= 'Hospital Data Science Project')
st.title('Interactive Statistical Analysis Dashboard')

# I used cache_data to prevent Streamlit from rereading the CSV file to improve speed.
@st.cache_data
def load_data():
    clean_df=pd.read_csv('clean_healthcare_dataset.csv')
    return clean_df
clean_df=load_data()
# I'm setting up the loading of the model


#Setting up the website titles and summart filters

st.subheader('Summary Tables Analysis')
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
var_list= [select_var1, select_var2]
if len(var_list) != len(set(var_list)):
    st.error('You cannot select the same variables for filtering. Please try again.')
    # I use this so that the other error from not defining filter_df is ignored.
    st.stop()
try:
    select =['Name', 'Age','Gender','Blood Type','Medical Condition', 'Date of Admission', 'Doctor', 'Hospital', 'Insurance Provider', 'Billing Amount ($)', 'Room Number', 'Admission Type', 'Discharge Date', 'Medication', 'Test Results', 'Days Spent']
    filter_row=clean_df[(clean_df[select_var1] >= range1[0]) & (clean_df[select_var1] <= range1[1])
                   &(clean_df[select_var2] >= range2[0])&(clean_df[select_var2] <= range2[1])]

    filter_df= filter_row[select]

except Exception as e:
    st.error(f'Table Error: {str(e)}')

if st.sidebar.button('I would like more options...'):
    str_cols1 = [
        col
        for col in clean_df.select_dtypes(include ='str').columns
        if col != 'Name'
    ]
    select_var3 =st.sidebar.selectbox('Select a string variable to display summary tables:', str_cols1)

    options = list(clean_df[select_var3].dropna().unique())
    range3 = st.sidebar.selectbox(f'Select value for {select_var3}:', options)
    var_list2 = [select_var1, select_var2, select_var3]
    if len(var_list2) != len(set(var_list2)):
            st.error('You cannot select the same variables for filtering. Please try again.')
# I use this so that the other error from not defining filter_df is ignored.
            st.stop()
    try:
            select =['Name', 'Age','Gender','Blood Type','Medical Condition', 'Date of Admission', 'Doctor', 'Hospital', 'Insurance Provider', 'Billing Amount ($)', 'Room Number', 'Admission Type', 'Discharge Date', 'Medication', 'Test Results', 'Days Spent']
            filter_row=clean_df[(clean_df[select_var1] >= range1[0]) & (clean_df[select_var1] <= range1[1])
                   &(clean_df[select_var2] >= range2[0])&(clean_df[select_var2] <= range2[1])
                   &(clean_df[select_var3] == range3)]
            filter_df= filter_row[select]
    except Exception as e:
            st.error(f'Table Error: {str(e)}')
# As I am using multiple 'and' statements, I seperate then by row to make it easier to read since I don't have cells like in Jupyter.

# Creating tabs
tab1, tab2, tab3 = st.tabs(['Summary Tables', 'Statistical Tests','Predictive Model'])

with tab1:
    
    st.subheader('Filtered Summary Tables')
    st.dataframe(filter_df,width= 'stretch')

with tab2:
    st.header('Statistical Tests')
    st.subheader('Anova Group Test amongst Company Groups')
    st.info('This Anova Test calculates if there is a significant difference in the Days Spent in their respective hospitals')
    
    if st.button('Press to see Anova Test'):
        anova_results = anova_load()
        st.markdown(anova_results)
    
    st.subheader('Distribution of Ages and Shapiro-Wilks Test')
    st.info('This shows the distribution of all ages in the dataset.')
    if st.button('General Normal Curve'):
        normal_fig= normal()
        st.pyplot(normal_fig)

    if st.button('Shapiro- Wilks Test'):
        shapy_test= shapiro()
        st.markdown(shapy_test)
        st.info("Note : From the value of the p-value, I assume that the data is too large for the Sharpiro-Wilks test to be accurate.According to the Central Limit Theorem, since N>5000 and it looks like a bell curve, I will assume the distribution of patient's age is normal so that I can use it for future tests as N=37144")

    st.subheader('Chi Squared Test')
    st.info('This Chi Squared Test is used to test if the Medical Condition and the Days Spent have any relationship.')
    if st.button('Generate Heatmap'):
            heat_fig= heat_map()
            st.pyplot(heat_fig)
            st.caption('Key: More concentrated colour indicates more patients')

with tab3:
    st.header('Predictor Model of the Billing Cost')
    st.warning('This model is trained on synthetic, inaccurate data. Do not use it to make informed decisions. Proceed with caution.')
    st.subheader('Enter Prediction Details:')
    assets= load_fullmodel()

    num_col, str_col = st.columns(2)
#df_features=['Age', 'Gender', 'Medication', 'Blood Type', 'Hospital', 'Days Spent','Admission Type Sort','Insurance Provider Sort','Medical Condition Sort','Test Results Sort']
    with num_col:
        age = st.number_input('Please enter your age:',1,100,50)
        gender= st.selectbox("Please enter your sex:" , ["Male","Female"])
        
        days= st.slider('How many Days did you spent in the hospital? (Select 1 if hospital visit was on the same day.):',1,365,5)
    
    with str_col:
        
       
        admin = st.selectbox('What would you say is your admission type?:', ['Elective','Urgent','Emergency','Other'])
        medic = st.selectbox('What is your Medical Condition?:', ['Diabetes','Arthritis','Obesity', 'Cancer',' Asthma', 'Hypertension','Other'])
        ins= st.selectbox('Who is your Insurance Provider?:', ['UnitedHealthcare','Cigna','Medicare','Blue Cross','Aetna', 'Other'])
        test= st.selectbox('What were your Test Results?:', ['Normal','Abnormal', 'Inconclusive', 'Other'])
    
#df_features=['Age', 'Gender', 'Days Spent','Admission Type Sort','Insurance Provider Sort','Medical Condition Sort','Test Results Sort']
    if st.button('Generate Prediction'):
        min_age= clean_df['Age'].min()
        max_age = clean_df['Age'].max()
        if age < min_age or age > max_age:
            st.warning(f'Warning:  The Age ({age}) chosen is outside the data range ({min_age} - {max_age}) Are you sure to want to proceed?')
        elif age is None:
            st.warning('You might want to enter a value for Age. Are you sure you want to proceed?')
        min_days= clean_df['Days Spent'].min()
        max_days= clean_df['Days Spent'].max()
        if days > max_days :
            st.warning(f'Warning :The number of days chosen ({days}) is outside the data range ({min_days}- {max_days}). Are you sure you want to proceed? ')
         
        admin_mapping={'Elective':0, 'Emergency':1, 'Urgent':2, 'Other':-1}
        admin_mapped=admin_mapping[admin]
        ins_mapping ={'UnitedHealthcare':0,'Cigna':1,'Aetna':2,'Blue Cross':3,'Medicare':4,'Humana':5,'Kaiser Permanente':6,'Anthem':7,'Centene':8,'Molina Healthcare':9, 'Other':-1}
        ins_mapped= ins_mapping[ins]
        medic_mapping ={'Hypertension':0, 'Asthma':1, 'Diabetes':2, 'Arthritis':3, 'Cancer':4, 'Heart Disease':5, 'Stroke':6, 'Kidney Disease':7, 'Liver Disease':8, 'Obesity':9, 'Depression':10, 'Anxiety':11, 'COPD':12, 'Osteoporosis':13, 'Alzheimer\'s Disease':14, 'Parkinson\'s Disease':15, 'Multiple Sclerosis':16, 'Epilepsy':17, 'HIV/AIDS':18, 'Other':-1}
        medic_mapped = medic_mapping[medic]
        test_mapping={'Normal':0, 'Abnormal':1, 'Other':-1}
        test_mapped= test_mapping[test]
        
        if admin_mapped or ins_mapped or medic_mapped == -1:
            st.warning("Please note that the predicted value will be less accurate and it will be predicted lower than it would, as 'Other' has been selected.")
        if gender == 'Female':
            gender_female = 1
            gender_male = 0
        else:
            gender_female = 0
            gender_male = 1
    
       
        try:
            final_features=pd.DataFrame(
                [{'Age':age,
                'Gender': gender, 
                'Days Spent' :days, 
                'Admission Type Sort':admin_mapped, 
                'Insurance Provider Sort':ins_mapped,
                'Medical Condition Sort':medic_mapped,
                'Test Results Sort' : test_mapped,}
                ])
            prediction = assets['rf_model'].predict(final_features)
            st.markdown('Final Estimation Cost')
            st.success(f'The Predicted Billing Amount is ${prediction[0]:.2f}')
            st.info(f'The MAE of this Model is around $12 000')
        except Exception as e:
            st.error(f'Transformation Error : {str(e)}')


    
        
        


    

    
    

  