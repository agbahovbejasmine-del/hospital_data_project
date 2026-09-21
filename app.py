import streamlit as st
#Setting up the website titles
st.set_page_config(page_title= 'Healthcare Data Engineering Project', layout='wide')
st.title('Healthcare Interactive Dashboard')
st.caption('An analytics platform evaluating a clinical dataset and a predictive billing model')

st.markdown('---')
st.header('Hospital Group Rankings by Performance')
try:
	df_rank =anova_data
	pass