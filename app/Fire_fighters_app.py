import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib

# Read data
# df = pd.read_csv("oct23_bds_int_firefighters\app\test_sample_i.csv")





# Sidebar title and page selection
st.sidebar.title("Table of Contents")
pages = ["Home","Introduction","Exploration", "Data Visualization", "Modelling"]
page = st.sidebar.radio("Go to", pages)

# Display content based on the selected page
if page == "Home":
    # Insert Introduction contents here
    # App title
    st.title("London Fire Brigade Response Time 2023") 
    st.image("Lego_1.jpg", caption="How quick we are at the incident?", use_column_width=True)  
    pass
elif page == "Introduction":
    # Insert Introduction contents here
    st.write("## Introduction")
    pass
elif page == "Exploration":
    # Insert Exploration contents here
    st.write("## Exploration")
    pass
elif page == "Data Visualization":
    # Insert DataViz contents here
    st.write("## Data Visualization")
    pass
elif page == "Modelling":
    # Insert Modelling contents here
    st.write("## Modelling")
    pass

