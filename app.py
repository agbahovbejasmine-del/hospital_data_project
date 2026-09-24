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

#num_cols3= clean_df.select_dtypes(include=[np.number]).columns.to_list()
#select_var3 =st.sidebar.selectbox('Select  third variable range to display summary tables:', num_cols3)
#min_value3=int(clean_df[select_var3].min())
#max_value3= int(clean_df[select_var3].max())
#range3= st.sidebar.slider('Select third value range:',min_value3, max_value3,(min_value3, max_value3)) 

# I'm using exceptions to prevent the user from using the same variables twice.

var_list= [select_var1, select_var2]
if len(var_list) != len(set(var_list)):
    st.error('You cannot select the same variables for filtering. Please try again.')
    # I use this so that the other error from not defining filter_df is ignored.
    st.stop()

# As I am using multiple 'and' statements, I seperate then by row to make it easier to read since I don't have cells like in Jupyter.
filter_df=clean_df[(clean_df[select_var1] >= range1[0]) & (clean_df[select_var1] <= range1[1])
                   &(clean_df[select_var2] >= range2[0])&(clean_df[select_var2] <= range2[1])]
                  # &(clean_df[select_var3] >= range3[0])&(clean_df[select_var3] <= range3[1])]


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
        age = st.number_input('Please enter your age:')
        gender= st.selectbox("Please enter your sex:" , ["Male","Female"])
        blood= st.selectbox("Please enter your blood type:", ['AB-','B-','A-','O+','A+','AB+','O-','B+'])
        days= st.slider('How many Days did you spent in the hospital?:')
    
    with str_col:
        hospital= st.text_input('What hospital do you go to?(Write with every first letter capitalized):')
        medication=st.selectbox('What medication are you using/do you use? :', ['Ibruprofen', 'Aspirin', 'Penicillin', 'Lipitor','Paracetamol','Other'])
        admin = st.selectbox('What would you say is your admission type?:', ['Elective','Urgent','Emergency','Other'])
        medic = st.selectbox('What is your Medical Condition?:', ['Diabetes','Arthritis','Obesity', 'Cancer',' Asthma', 'Hypertension','Other'])
        ins= st.selectbox('Who is your Insurance Provider?:', ['UnitedHeathcare','Cigna','Medicare','Blue Cross','Aetna', 'Other'])
        test= st.selectbox('What were your Test Results?:', ['Normal','Abnormal', 'Inconclusive', 'Other'])
    

    if st.button('Generate Prediction'):
        min_age= clean_df['Age'].min()
        max_age = clean_df['Age'].max()
        if age < min_age or age > max_age:
            st.warning(f'Warning: {age} is outside the data range ({min_age - max_age}) Are you sure to want to proceed?')
        elif age is None:
            st.warning('You might want to enter a value for Age. Are you sure you want to proceed?')
         
        admin_mapping={'Elective':0, 'Emergency':1, 'Urgent':2}
        admin_mapped=admin_mapping[admin]
        ins_mapping ={'UnitedHealthcare':0,'Cigna':1,'Aetna':2,'Blue Cross':3,'Medicare':4,'Humana':5,'Kaiser Permanente':6,'Anthem':7,'Centene':8,'Molina Healthcare':9}
        ins_mapped= ins_mapping[ins]
        medic_mapping ={'Hypertension':0, 'Asthma':1, 'Diabetes':2, 'Arthritis':3, 'Cancer':4, 'Heart Disease':5, 'Stroke':6, 'Kidney Disease':7, 'Liver Disease':8, 'Obesity':9, 'Depression':10, 'Anxiety':11, 'COPD':12, 'Osteoporosis':13, 'Alzheimer\'s Disease':14, 'Parkinson\'s Disease':15, 'Multiple Sclerosis':16, 'Epilepsy':17, 'HIV/AIDS':18}
        medic_mapped = medic_mapping[medic]
        test_mapping={'Normal':0, 'Abnormal':1}
        test_mapped= test_mapping[test]
       
       if gender =='Female':
            gender_female= 1
            gender_male= 0
            
       else:
            gender_female= 1
            gender_male= 0
    
       
        try:
            
            if hasattr(encoded_features, 'toarray'):
                encoded_features = encoded_features.toarray()
            final_features=np.hstack([age, gender_mapped, days, admin_mapped, ins_mapped, medic_mapped, test_mapped]).reshape(1,-1)
            prediction = assets['model'].predict(final_features)
            st.markdown('Final Estimation Cost')
            st.success(f'The Predicted Billing Amount is ${prediction :.2f}')
            st.info(f'The MAE of this Model is around $12 000')
        except Exception as e:
            st.error(f'Transformation Error : {str(e)}')


    
        
        


    

    
    

  