import streamlit as st
import pandas as pd
import plotly.express as px

# Function to load data from an Excel file
def load_data(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name=sheet_name)

# Load data from three different Excel sheets
data1 = load_data('BBB.xlsx', 'Sheet1')
data2 = load_data('google.xlsx', 'Sheet1')
data3 = load_data('ipata.xlsx', 'Sheet1')

st.title('SMART Pet Air Travel Market Comparative Dashboard')

# Summarize Google review data into visualization
google_data = data2

# Convert Rating to numeric and handle NaN values
google_data['Rating'] = pd.to_numeric(google_data['Rating'].str.extract(r'(\d+)')[0], errors='coerce').fillna(0)

# Group by Name, Category, and Address to get the amount of reviews and average rating
summary = google_data.groupby(['Name', 'Category', 'Address']).agg(
    amount_of_reviews=('Rating_count', 'sum'),
    average_rating=('Rating', 'mean')
).reset_index()

# Create a scatter plot with hover information
fig = px.scatter(summary, x='amount_of_reviews', y='average_rating', color='Category',
                 hover_data=['Name', 'Address'],
                 labels={'amount_of_reviews': 'Amount of Reviews', 'average_rating': 'Average Rating'},
                 title='Google Review Summary', width=1000, height=800)


st.plotly_chart(fig)