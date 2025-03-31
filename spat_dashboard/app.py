import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from collections import Counter
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta


# =============================================================================
# Header and Competitor Website Analysis Section
# =============================================================================
st.title("SMART Pet Air Travel Competitive Marketing Dashboard")
st.write("This section loads competitor data from BBB, extracts geographic"
         " details from addresses, merges it with US states data for geolocation,"
         " and creates visualizations of competitor distribution on a US map and a bar chart of top services.")

# Competitor Website URL Input
url = st.text_input("Enter Competitor's Website URL")
if st.button("Gather Insights"):
    if url:
        st.write("Analyzing the website...")  # Placeholder for web scraping/insights logic


# =============================================================================
# BBB Data Analysis: Competitor Data & Visualizations
# =============================================================================
@st.cache_data
def load_bbb_data():
    pet_bbb = pd.read_csv('BBB_Clean_Octoparse.csv', encoding='latin1')
    # Rename columns
    pet_bbb.columns = ['Company_Name', 'Company_Address', 'BBB_Rating', 'Phone',
                       'BusinessStarted', 'NumberEmployees', 'BusinessCategories']
    return pet_bbb


def parse_address(address):
    # Attempt to extract City, State and Postal Code from the address string
    try:
        parts = [part.strip() for part in address.split(',')]
        city = parts[0] if len(parts) > 0 else None
        state = parts[-1].split(' ')[0] if len(parts) > 1 else None
        postal_code = parts[-1].split(' ')[1] if len(parts) > 1 and len(parts[-1].split(' ')) > 1 else None
        return city, state, postal_code
    except:
        return None, None, None


def process_bbb_data(bbb_df):
    # Parse address into separate columns
    bbb_df[['City', 'State', 'Postal_code']] = bbb_df['Company_Address'].apply(
        lambda x: pd.Series(parse_address(x))
    )
    # Process Business Categories
    bbb_df['BusinessCategories'] = bbb_df['BusinessCategories'].apply(
        lambda x: x.split(',') if pd.notnull(x) else []
    )
    bbb_df['NumberOfServices'] = bbb_df['BusinessCategories'].apply(len)
    return bbb_df


pet_bbb = load_bbb_data()
pet_bbb = process_bbb_data(pet_bbb)


# Merge with US States to get geocoordinates for mapping
@st.cache_data
def load_us_states():
    return pd.read_csv('US_States.csv')


us_states = load_us_states()
pet_bbb = pd.merge(pet_bbb, us_states, on='State', how='left')

# Competitor selection from BBB Data
unique_companies = tuple(pet_bbb['Company_Name'].unique().tolist())
selected_company = st.selectbox('Select the name of a pet company', unique_companies)
st.write('You selected:', selected_company)


# Visualizations: US Map and Top Services
def usmap_distribution():
    st.subheader("Distribution of Competitors Across the US")
    # Plotly scatter_geo map using LAT and LON from the US_States dataset
    fig = px.scatter_geo(pet_bbb, lat="LAT", lon="LON", hover_name="Company_Name",
                         hover_data=['BBB_Rating', 'NumberEmployees'], scope="usa")
    st.plotly_chart(fig)


usmap_distribution()

# Top services bar chart
pet_rows = pet_bbb.explode('BusinessCategories')
pet_rows['BusinessCategories'] = pet_rows['BusinessCategories'].apply(lambda x: x.strip() if pd.notnull(x) else None)
df_top_services = pet_rows.groupby('BusinessCategories')['Company_Name'].nunique().sort_values(
    ascending=False).reset_index().head(10)
df_top_services.columns = ['Services', 'Number_of_companies']


def top_services(df_top_services):
    st.subheader("Top Services Provided by Various Companies")
    fig = px.bar(df_top_services, x="Services", y="Number_of_companies",
                 height=400,
                 title='Number of Companies Providing Top 10 Services')
    st.plotly_chart(fig)


top_services(df_top_services)

# =============================================================================
# Competitive Marketing Dashboard (Company Summary & Google Reviews)
# =============================================================================
st.header("Compare and Analyze Company Sentiment")
st.write("This section displays company-level data, review-level sentiment and categorizations, "
         "and correlations between review ratings and positive mentions from Google Reviews.")


# -------------------------
# Data Loading Functions
# -------------------------
@st.cache_data
def load_company_data(file_path):
    df = pd.read_excel(file_path)
    # Convert Strengths and Weaknesses from string representations to dictionaries
    strengths = df["Strengths"].apply(eval)
    weaknesses = df["Weaknesses"].apply(eval)
    return df, strengths, weaknesses


@st.cache_data
def load_google_reviews(file_path):
    google_reviews = pd.read_excel(file_path)
    # Remove " stars", strip spaces, and convert to numeric
    google_reviews['Rating'] = pd.to_numeric(
        google_reviews['Rating']
        .str.replace(" stars", "", regex=False)
        .str.strip(),
        errors='coerce'
    )
    return google_reviews.dropna(subset=['Rating'])


df, strengths, weaknesses = load_company_data("company_summary.xlsx")
google_reviews = load_google_reviews("google.xlsx")


# -------------------------
# Data Preparation Functions
# -------------------------
def prepare_viz_data(df, strengths, weaknesses, categories):
    """
    Create a DataFrame with one row per (company, category) along with strength and weakness values.
    """
    viz_data = []
    for company, strength, weakness in zip(df["Company"], strengths, weaknesses):
        for category in categories:
            viz_data.append({
                "Company": company,
                "Category": category,
                "Strengths": strength.get(category, 0),
                "Weaknesses": weakness.get(category, 0)
            })
    return pd.DataFrame(viz_data)


categories_list = ["Customer Service", "Pricing", "Communication",
                   "Ease of Process", "Safety and Care", "Reputation"]
viz_df = prepare_viz_data(df, strengths, weaknesses, categories_list)


# -------------------------
# Visualization Functions
# -------------------------
def plot_comparison(viz_df, selected_companies):
    filtered_df = viz_df[viz_df["Company"].isin(selected_companies)]
    strengths_fig = px.bar(filtered_df, x="Category", y="Strengths", color="Company",
                           barmode="group", title="Strengths per Category for Selected Companies")
    st.plotly_chart(strengths_fig)

    weaknesses_fig = px.bar(filtered_df, x="Category", y="Weaknesses", color="Company",
                            barmode="group", title="Weaknesses per Category for Selected Companies")
    st.plotly_chart(weaknesses_fig)


def plot_google_reviews_scatter(google_reviews, selected_companies):
    aggregated_reviews = google_reviews.groupby("Name").agg(
        Avg_Rating=("Rating", "mean"),
        Review_Count=("Rating", "size")
    ).reset_index()
    if "All" in selected_companies:
        filtered_reviews = aggregated_reviews
    else:
        filtered_reviews = aggregated_reviews[aggregated_reviews["Name"].isin(selected_companies)]
    scatter_fig = px.scatter(filtered_reviews, x="Avg_Rating", y="Review_Count", color="Name",
                             title="Average Ratings vs. Rating Count for Selected Companies",
                             labels={"Avg_Rating": "Average Rating (Stars)", "Review_Count": "Total Number of Ratings"},
                             size_max=30)
    st.plotly_chart(scatter_fig)


# -------------------------
# Sentiment & Review-Level Correlation Functions
# -------------------------
categories_dict = {
    "Customer Service": ["responsive", "helpful", "supportive", "service", "team"],
    "Pricing": ["affordable", "expensive", "cost", "pricing", "value"],
    "Communication": ["updates", "informative", "contact", "communicative"],
    "Ease of Process": ["easy", "smooth", "process", "hassle-free", "booking"],
    "Safety and Care": ["safe", "care", "well-being", "condition"],
    "Reputation": ["reliable", "recommend", "trustworthy", "professional", "reputation"],
}


def analyze_sentiment(review):
    if not isinstance(review, str):
        review = str(review)
    blob = TextBlob(review)
    sentiment_score = blob.sentiment.polarity  # Range: -1 (negative) to +1 (positive)
    return "Positive" if sentiment_score > 0 else "Negative"


def categorize_review(review):
    if not isinstance(review, str):
        review = str(review)
    review_categories = []
    for category, keywords in categories_dict.items():
        if any(keyword in review.lower() for keyword in keywords):
            review_categories.append(category)
    return review_categories


def create_detailed_review_summary(df):
    df = df.copy()
    df["Sentiment"] = df["Review"].apply(analyze_sentiment)
    df["Categories"] = df["Review"].apply(categorize_review)
    return df


def create_company_review_summary(google_reviews):
    google_reviews["Review"] = google_reviews["Review"].astype(str)
    if "Categories" not in google_reviews.columns:
        google_reviews["Categories"] = google_reviews["Review"].apply(categorize_review)
    summary_data = []
    for company, group in google_reviews.groupby("Name"):
        all_cats = []
        for cats in group["Categories"]:
            all_cats.extend(cats)
        cat_counter = Counter(all_cats)
        summary_str = ", ".join([f"{cat} ({cnt})" for cat, cnt in
                                 cat_counter.most_common()]) if cat_counter else "No categories identified."
        summary_data.append({"Company": company, "Review Summary": summary_str})
    return pd.DataFrame(summary_data)


# -------------------------
# Main Dashboard: Company & Review Visualizations
# -------------------------
all_companies = list(viz_df["Company"].unique()) + list(google_reviews["Name"].unique())
selected_companies = st.multiselect("Choose one or more companies to visualize:",
                                    options=set(all_companies),
                                    default=None)

if selected_companies:

    st.subheader("Google Reviews: Ratings Distribution")
    plot_google_reviews_scatter(google_reviews, selected_companies)

    st.subheader("Strengths and Weaknesses Summary")
    plot_comparison(viz_df, selected_companies)

    review_summary_df = create_company_review_summary(google_reviews)
    st.dataframe(review_summary_df)

    # -------------------------
    # Review-Level Correlation Analysis
    # -------------------------
    # Create a detailed review summary if needed (e.g., for further analysis)
    detailed_review_summary = create_detailed_review_summary(google_reviews)

    # Explode the categories for each review to analyze correlations
    detailed_exploded = detailed_review_summary.explode("Categories")
    detailed_exploded["Rating"] = pd.to_numeric(detailed_exploded["Rating"], errors="coerce")

    category_correlations = {}
    for cat in categories_dict.keys():
        detailed_exploded["Cat_Strength_Flag"] = detailed_exploded.apply(
            lambda row: 1 if (row["Sentiment"] == "Positive" and row["Categories"] == cat) else 0, axis=1
        )
        corr_value = detailed_exploded["Rating"].corr(detailed_exploded["Cat_Strength_Flag"])
        category_correlations[cat] = corr_value

    corr_df_new = pd.DataFrame(list(category_correlations.items()), columns=["Category", "Correlation"])
    st.subheader("Correlation between Review Ratings and Positive Mentions by Category")
    st.write("""
        ## Key Insights:
        - **Overall Strength Impact**: A stronger overall company strength often correlates with better ratings.
        - **Review-Level Analysis**: The correlation values indicate how positive mentions in specific categories relate to ratings.
    """)
    fig_corr = px.bar(corr_df_new, x="Category", y="Correlation",
                      title="Review Rating vs. Positive Category Mentions Correlation",
                      labels={"Correlation": "Correlation Value", "Category": "Review Category"},
                      color="Correlation", color_continuous_scale="RdBu")
    st.plotly_chart(fig_corr)

    # -------------------------
    # Google Reviews Timeline Visualization
    # -------------------------
    if "Review_time" in google_reviews.columns:
        st.subheader("Google Reviews Timeline by Company")


        def parse_review_time(time_str):
            """
            Converts a relative time string into an approximate datetime object.
            """
            time_str = str(time_str).lower().strip()
            now = datetime.now()
            # Handle "a month ago" or "a year ago"
            if time_str.startswith("a month"):
                return now - relativedelta(months=1)
            if time_str.startswith("a year"):
                return now - relativedelta(years=1)
            # Handle "X months ago"
            months_match = re.search(r"(\d+)\s+months?\s+ago", time_str)
            if months_match:
                months = int(months_match.group(1))
                return now - relativedelta(months=months)
            # Handle "X years ago"
            years_match = re.search(r"(\d+)\s+years?\s+ago", time_str)
            if years_match:
                years = int(years_match.group(1))
                return now - relativedelta(years=years)
            # Fallback: return current date
            return now


        # Parse review times into an approximate date
        google_reviews["Approx_Date"] = google_reviews["Review_time"].apply(parse_review_time)

        # Filter for selected companies only
        timeline_df = google_reviews[google_reviews["Name"].isin(selected_companies)].copy()

        # For each company, sort by date and compute the cumulative count of reviews
        timeline_df = timeline_df.sort_values(by=["Name", "Approx_Date"])
        timeline_df["Cumulative_Reviews"] = timeline_df.groupby("Name").cumcount() + 1

        # Create a multi-line chart with Plotly Express
        st.write("### Cumulative Number of Reviews Over Time")
        fig_line = px.line(timeline_df, x="Approx_Date", y="Cumulative_Reviews", color="Name",
                           title="Cumulative Reviews Timeline by Company",
                           labels={"Approx_Date": "Date", "Cumulative_Reviews": "Cumulative Reviews",
                                   "Name": "Company"})
        st.plotly_chart(fig_line)

        # Reviews per Month grouped by company
        timeline_df["Month_Year"] = timeline_df["Approx_Date"].dt.to_period("M").astype(str)
        reviews_per_month = timeline_df.groupby(["Name", "Month_Year"])["Rating"].count().reset_index()
        reviews_per_month.columns = ["Name", "Month_Year", "Reviews"]
        reviews_per_month = reviews_per_month.sort_values(by=["Name", "Month_Year"])

        st.write("### Reviews per Month by Company")
        fig_bar = px.bar(reviews_per_month, x="Month_Year", y="Reviews", color="Name",
                         barmode="group",
                         title="Monthly Reviews by Company",
                         labels={"Month_Year": "Month-Year", "Reviews": "Number of Reviews", "Name": "Company"})
        st.plotly_chart(fig_bar)
    else:
        st.info("The Google Reviews data does not contain a 'Review_time' column for timeline visualization.")
else:
    st.warning("Please select at least one company to visualize!")
