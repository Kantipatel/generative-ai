import streamlit as st
import pandas as pd
import numpy as np
import os

cwd = os.getcwd()
file_name = f"{cwd}/../data/model_types.csv"
df = pd.read_csv(file_name)
   

st.title('Custom ChatGPT App')
st.text('This is a custom app for ChatGPT built using streamlit.\n')
st.text('Models to be supported, work in progress\n')
#TODO work in progress:
# show user input box for query, selection options, clear, and submit buttons
# call remote API to get response
st.write(df.head(10))
