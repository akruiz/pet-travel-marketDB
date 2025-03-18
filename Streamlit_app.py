import streamlit as st
from streamlit.web import cli as stcli
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

st.title("Competitive Marketing Analysis for Smart Pet Air Travel (SPAT)")
url = st.text_input("Enter Competitor's Website URL")
if st.button("Gather Insights"):
    if url:
        st.write("Analyzing the website...")



def state_split(x):
    try:
        return x.split(',')[-2].lstrip().split(' ')[0]
    except:
        return None
    

def postal_code_split(x):
    try:
        return x.split(',')[-1].lstrip().split(' ')[1]
    except:
        return None

def city_split(x):
    x_list = x.split(',')
    if len(x_list) >= 3:
        try:
            return x_list[-(len(x_list))]
        except:
            return None
    else: 
        return None
    
pet_bbb = pd.read_csv('BBB_Clean_Octoparse.csv')
print(pet_bbb.shape)

pet_bbb.columns = ['Company_Name', 'Company_Address',  'BBB_Rating',  'Phone' ,'BusinessStarted', 'NumberEmployees', 'BusinessCategories']

unique_companies = tuple(pet_bbb['Company_Name'].unique().tolist())
option = st.selectbox(
'Select the name of pet company', unique_companies)

st.write('You selected:', option)


def state_split(x):
    try:
        return x.split(',')[-1].lstrip().split(' ')[0]
    except:
        return None
    

def postal_code_split(x):
    try:
        return x.split(',')[-1].lstrip().split(' ')[1]
    except:
        return None

def city_split(x):
    x_list = x.split(',')
    if len(x_list) >= 3:
        try:
            return x_list[-(len(x_list)-1)]
        except:
            return None
    else: 
        return None
    
pet_bbb['City'] = pet_bbb['Company_Address'].apply(lambda x: city_split(x))
pet_bbb['State'] = pet_bbb['Company_Address'].apply(lambda x: state_split(x))
pet_bbb['Postal_code'] = pet_bbb['Company_Address'].apply(lambda x: postal_code_split(x))

# pet_bbb['NumberEmployees'].fillna('NA', inplace = True)

pet_bbb['BusinessCategories'] = pet_bbb['BusinessCategories'].apply(lambda x: x.split(',') if pd.notnull(x) else None)
pet_bbb['NumberOfServices'] = pet_bbb['BusinessCategories'].apply(lambda x: len(x) if(np.all(pd.notnull(x))) else None)

## Reading the US states
us_states = pd.read_csv('US_States.csv')
pet_bbb = pd.merge(pet_bbb, us_states, on = 'State', how = 'left')


def usmap_distribution():
    
    st.subheader("Distribution of comeptitors accross the US")
    # Create the Plotly map
    fig = px.scatter_geo(pet_bbb, lat="LAT", lon="LON", hover_name="Company_Name" ,hover_data = ['BBB_Rating', 'NumberEmployees'], scope="usa")

    # Display the map in Streamlit
    st.plotly_chart(fig)
usmap_distribution()


pet_rows =  pet_bbb.explode('BusinessCategories')
pet_rows['BusinessCategories'] = pet_rows['BusinessCategories'].apply(lambda x: x.strip() if pd.notnull(x) else None)
df_top_services = pet_rows.groupby('BusinessCategories')['Company_Name'].nunique().sort_values(ascending = False).reset_index().head(10)
df_top_services.columns = ['Services', 'Number_of_companies']

def top_services(df_top_services):

    st.subheader("Top Services provided by various companies")

    fig = px.bar(df_top_services, x="Services", y="Number_of_companies", 
                height=400,
                title='Number of companies providing top 10 services')
    st.plotly_chart(fig)

top_services(df_top_services)