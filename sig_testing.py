# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 14:04:27 2021

@author: bmussa
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import streamlit as st

#%%

st.set_page_config(layout='wide')

st.title("Significance Testing App")
st.write("In this app I am using Mann-whitney U test to test the significance for data used in an A/B test"
         " as the data isnt normally distributed."
         " I will also create confidence intervals using bootstrapping"
         )

st.sidebar.header('Upload your CSV data')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
st.sidebar.markdown("""
[Example CSV input file]
(https://raw.githubusercontent.com/bilalmussa/st_sig_testing/main/sig_test_file.csv)
""")

reps = st.sidebar.text_input("Enter the number of repetitions to create CI's",100)
sig_level = st.sidebar.text_input("Enter your level of significance in decimal places",0.10)


@st.cache(suppress_st_warning=True,allow_output_mutation=True)
def tidy_data(file_path):
    df = pd.read_csv(file_path)
    df = df.fillna(0)
    return df

#uploaded_df= tidy_data(uploaded_file)
if not uploaded_file:
    uploaded_df = tidy_data('https://raw.githubusercontent.com/asad-mahmood/66DaysOfData/main/Heart%20Failure/heart_failure_clinical_records_dataset.csv')
else:
    uploaded_df= tidy_data(uploaded_file)

st.write(uploaded_df[['Test','Control']].describe()[:2])

#%% perform the Mann-Whitney U test

group1 = uploaded_df['Test'].to_list()
group2 = uploaded_df['Control'].to_list()
s, p = stats.mannwhitneyu(group1, group2, alternative='two-sided')
#st.write(stats.mannwhitneyu(group1, group2, alternative='two-sided'))
st.write("Statistc: ",s)
st.write("P Value: ",p)

if p<0.01:
    st.write("This is significant @ 1% level of significance")
elif p<0.05:
    st.write("This is significant @ 5% level of significance")
elif p<0.1:
    st.write("This is significant @ 10% level of significance")
else:
    st.write("Not significant at any conventional level")
    
#%%
def bootstrap_ci(df, variable, classes, repetitions = 1000, alpha = 0.05, random_state=None): 
    df = df[[variable, classes]]
    bootstrap_sample_size = len(df) 
    
    mean_diffs = []
    for i in range(repetitions):
        bootstrap_sample = df.sample(n = bootstrap_sample_size, replace = True, random_state = random_state)
        mean_diff = bootstrap_sample.groupby(classes).mean().iloc[1,0] - bootstrap_sample.groupby(classes).mean().iloc[0,0]
        mean_diffs.append(mean_diff)
            # confidence interval
        left = np.percentile(mean_diffs, alpha/2*100)
        right = np.percentile(mean_diffs, 100-alpha/2*100)
    # point estimate
    point_est = df.groupby(classes).mean().iloc[1,0] - df.groupby(classes).mean().iloc[0,0]
    st.write('Point estimate of difference between means:', round(point_est,2))
    st.write((1-alpha)*100,'%','confidence interval for the difference between means:', (round(left,2), round(right,2)),' this is base on ',repetitions, ' repetitions')
    
    
df = uploaded_df[['Test','Control']].stack().reset_index().rename(columns={'level_0':'row','level_1':'cut', 0:'dp'})

bootstrap_ci(df,'dp','cut',int(reps),float(sig_level))
