Hospital Data Engineering and Analysis Pipeline:
 - Cleaned and processed data from a large dataset consisting of 37144 patient records, and formatting and creating relevant summary tables to use for future development.
 - Using a range of statistical tests to analyse patient data from a Kaggle dataset, such as Chi-squared tests, an ANOVA test, a Shapiro-Wilks Test, and creating a Matplotlib chart on the normal distribution of patients' ages.
 - Created two predictive models for the billing amount of patients' treatment , based on factors such as their Age, Gender , Insurance Provider etc. This has also been used in the backend to compare the accuracy of these models based on their Mean Absolute Error.
 - Did a Frontend using Streamlit to create a summary table that can be filtered based on variables, with statistics per column, a summary of my tests I did on Jupyter, and finally the predictive model.
CONCLUSION
-This Project was carried out to sort the messy Kaggle Dataset into meaningful and easy-to-read information. I also did the statistical tests to see if it stacks up against real life data.
-The tests carried out (such as the test between the blood type and the Medical Condition) proved that the data showed no meaningful correlation between things that would have a more significant correlation in reality (i.e. Medical Condition and Days Spent). Also, the supposably less significant correlation scored better than the other in terms of correlation in the Chi Squared Test.
-However, the Age distribution in the data was normal, which could be expected in real hospital data.
-The models also had significantly high MAEs and MSEs, which would be much lower with a dataset that consisted of real patient data. They also did not change greatly when I implemented the appropriate encoders to the features.
-I therefore concluded that the dataset must have been synthetically created due to these results.
