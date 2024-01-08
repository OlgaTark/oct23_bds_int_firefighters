import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
from PIL import Image

# Read data
# df = pd.read_csv("oct23_bds_int_firefighters\app\test_sample_i.csv")
df_i_2022=pd.read_csv("oct23_bds_int_firefighters/data/raw/df_i_2022.csv")





# Sidebar title and page selection
st.sidebar.title("Table of Contents")
pages = ["Home","Introduction","Exploration", "Data Visualization", "Modelling"]
page = st.sidebar.radio("Go to", pages)

# Display content based on the selected page
if page == "Home":
    # Insert Introduction contents here
    # App title
    st.title("London Fire Brigade Response Time 2023") 
    st.image("oct23_bds_int_firefighters/app/Lego_1.jpg", caption="How quick we are at the incident?", use_column_width=True)  
    pass
elif page == "Introduction":
    # Insert Introduction contents here
    st.write("## Introduction")
    pass
elif page == "Exploration":
    # Insert Exploration contents here
    st.write("## Exploration")
    st.dataframe(df_i_2022.head(10))
    st.write(df_i_2022.shape)
    st.dataframe(df_i_2022.describe())
  
    if st.checkbox("Show NA") :
       st.dataframe(df_i_2022.isna().sum())
    pass
elif page == "Data Visualization":
    # Insert DataViz contents here
    st.write("## Data Visualization")
    st.image("oct23_bds_int_firefighters/app/Crosstab_meantime.png", use_column_width=True)
    pass
elif page == "Modelling":
    # Insert Modelling contents here
    st.write("## Modelling")
    pass

