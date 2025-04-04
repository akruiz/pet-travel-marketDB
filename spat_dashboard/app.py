import plotly.express as px
from textblob import TextBlob
from collections import Counter
from datetime import datetime
from dateutil.relativedelta import relativedelta
import streamlit as st
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from google.oauth2 import service_account
from google.cloud import bigquery

# Service account key file
key_file = 'spat-455404-8aa716b59add.json'  # Update the JSON file name here
scopes = ["https://www.googleapis.com/auth/cloud-platform",
          "https://www.googleapis.com/auth/drive"]
credentials = service_account.Credentials.from_service_account_file(key_file, scopes=scopes)

client = bigquery.Client(credentials=credentials, project="spat-455404")
project_id = "spat-455404"
dataset_id = "SPAT"
tables = ["BBB", "GoogleReview", "IPATA", "Map_Pricing"]

# Fetch data
df_dict = {}
for table in tables:
    query = f"SELECT * FROM `{project_id}.{dataset_id}.{table}`"
    df1 = client.query(query).to_dataframe()
    df_dict[table] = df1.iloc[1:]  # Skip the first row because it's duplicated in the Google Sheet

BBB_df = df_dict.get("BBB", pd.DataFrame())
GoogleReview_df = df_dict.get("GoogleReview", pd.DataFrame())
IPATA_df = df_dict.get("IPATA", pd.DataFrame())
MapPricing_df = df_dict.get("Map_Pricing", pd.DataFrame())


# =============================================================================
# Helper Functions for Competitor Insights
# =============================================================================
def calculate_seo_score(soup):
    """
    Compute a basic SEO score based on on-page attributes
    """
    score = 0
    # -- Title Tag Check
    title_tag = soup.find("title")
    if title_tag and title_tag.get_text(strip=True):
        score += 20
    # -- Meta Description Check
    meta_desc_tag = soup.find("meta", attrs={"name": "description"})
    if meta_desc_tag and meta_desc_tag.get("content"):
        meta_desc_content = meta_desc_tag.get("content").strip()
        if 50 <= len(meta_desc_content) <= 160:
            score += 20
        else:
            score += 10
    # -- H1 Tag Check
    h1_tags = soup.find_all("h1")
    if len(h1_tags) == 1:
        score += 20
    elif len(h1_tags) > 1:
        score += 10
    # -- Image ALT Attribute Check
    images = soup.find_all("img")
    if images:
        images_with_alt = sum(1 for img in images if img.get("alt"))
        percentage_with_alt = (images_with_alt / len(images)) * 100
        if percentage_with_alt > 90:
            score += 20
        elif percentage_with_alt > 70:
            score += 10
    # -- Canonical Link Check
    canonical_tag = soup.find("link", attrs={"rel": "canonical"})
    if canonical_tag:
        score += 10
    return score


def extract_years_in_business(full_text):
    """
    Attempts to find the founding year and computes how many years in business.
    """
    current_year = datetime.now().year
    year_match = re.search(r'(?i)(?:established|founded|since)\s+(?:in\s+)?((?:19|20)\d{2})', full_text)
    if year_match:
        founding_year = int(year_match.group(1))
        years = current_year - founding_year
        return f"{years} years in business (Founded in {founding_year})"
    over_years = re.search(r'over\s+(\d+)\s+years', full_text, re.I)
    if over_years:
        return f"Over {over_years.group(1)} years of experience"
    return "No clear founding year or years in business information found."


def extract_animal_services(full_text):
    """
    Searches the text for common animals to determine types of services.
    """
    animals_keywords = [
        "dog", "dogs", "cat", "cats", "bird", "birds",
        "reptile", "reptiles", "fish", "hamster", "hamsters",
        "rabbit", "rabbits", "exotic"
    ]
    found_animals = set()
    lower_text = full_text.lower()
    for animal in animals_keywords:
        if animal in lower_text:
            found_animals.add(animal)
    return list(found_animals) if found_animals else "No specific animal services mentioned."

def extract_pricing_info(full_text):
    """
    Searches for pricing information in the text by checking each sentence.
    """
    pricing_keywords = [
        "price", "cost", "payment", "pay", "charge", "pricing", "expensive",
        "affordable", "flexible", "varies", "custom quote", "by appointment",
        "contact for pricing", "get a quote", "quote"
    ]

    # Regular expression for common dollar amount patterns
    dollar_pattern = re.compile(r'\$\s*\d+(?:[,\d]*)(?:\.\d+)?')

    # Split text into sentences (this can be refined further if needed)
    sentences = re.split(r'(?<=[.!?])\s+', full_text)

    found_sentences = set()
    for sentence in sentences:
        # Check if sentence contains a dollar amount
        if dollar_pattern.search(sentence):
            found_sentences.add(sentence.strip())
        else:
            # Check if any pricing keyword is in the sentence
            lower_sentence = sentence.lower()
            if any(keyword in lower_sentence for keyword in pricing_keywords):
                found_sentences.add(sentence.strip())

    if found_sentences:
        return list(found_sentences)
    else:
        return "No pricing information available."


def extract_tech_stack(soup, full_text):
    """
    Attempts to identify the website stack by searching for common platform indicators.
    """
    stack_indicators = {
        "WordPress": ["wp-content", "wordpress"],
        "Shopify": ["shopify"],
        "Wix": ["wix"],
        "Squarespace": ["squarespace"],
        "Magento": ["magento"],
        "Drupal": ["drupal"]
    }
    found_stack = []
    for tech, indicators in stack_indicators.items():
        for indicator in indicators:
            if indicator in full_text.lower():
                found_stack.append(tech)
                break
    # Check for a meta generator tag
    meta_generator = soup.find("meta", attrs={"name": "generator"})
    if meta_generator and meta_generator.get("content"):
        found_stack.append(meta_generator.get("content"))
    return list(set(found_stack)) if found_stack else "No website stack information detected."


def gather_competitor_insights(url):
    """
    Retrieves the website content and extracts various insights including SEO, business details,
    website stack, pricing, and other service information.
    """
    insights = {}
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        insights["error"] = f"Error fetching the website: {e}"
        return insights
    soup = BeautifulSoup(response.content, 'html.parser')
    full_text = soup.get_text(" ", strip=True)

    # -- Extract Description
    meta_desc_tag = soup.find("meta", attrs={"name": "description"})
    if meta_desc_tag and meta_desc_tag.get("content"):
        description = meta_desc_tag.get("content").strip()
    else:
        paragraphs = soup.find_all("p")
        para_texts = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
        description = " ".join(para_texts[:2]) if para_texts else "No description available."
    insights["description"] = description

    # -- Extract Service Information
    industry_keywords = [
        "pet", "shipping", "service", "transport", "care", "veterinary",
        "safe", "cruelty-free", "specialized", "custom"
    ]
    potential_info = []
    for tag in soup.find_all(["h1", "h2", "h3", "p"]):
        text = tag.get_text(" ", strip=True)
        if any(kw in text.lower() for kw in industry_keywords) and len(text) > 20:
            potential_info.append(text)
    seen = set()
    unique_info = []
    for info in potential_info:
        if info not in seen:
            unique_info.append(info)
            seen.add(info)
    insights["services_and_products"] = unique_info if unique_info else [
        "No clear service or product details detected."]

    # -- Calculate SEO Score
    insights["seo_score"] = calculate_seo_score(soup)
    # -- Extract Years in Business
    insights["years_in_business"] = extract_years_in_business(full_text)
    # -- Extract Animal Services
    insights["animal_services"] = extract_animal_services(full_text)
    # -- Extract Pricing Information
    insights["pricing_info"] = extract_pricing_info(full_text)
    # -- New Extraction: Website Stack
    insights["tech_stack"] = extract_tech_stack(soup, full_text)

    return insights


# =============================================================================
# Streamlit Dashboard Layout
# =============================================================================

# Two-sentence overview for the tool
st.title("SMART Pet Air Travel Competitive Marketing Dashboard")
st.write(
    ""
)


# ---------------------------
# BBB Data Analysis & Visualizations Section
# ---------------------------

@st.cache_data
def load_bbb_data():
    # commented if you want to run locally uncomment and comment df pull
    #pet_bbb = pd.read_csv('BBB_Clean_Octoparse.csv', encoding='latin1')
    pet_bbb = BBB_df
    pet_bbb.columns = ['Company', 'Address', 'BBBRating', 'Phone',
                       'BusinessStarted', 'NumberEmployees', 'BusinessCategories']
    return pet_bbb


def parse_address(address):
    try:
        parts = [part.strip() for part in address.split(',')]
        city = parts[0] if len(parts) > 0 else None
        state = parts[-1].split(' ')[0] if len(parts) > 1 else None
        postal_code = parts[-1].split(' ')[1] if len(parts) > 1 and len(parts[-1].split(' ')) > 1 else None
        return city, state, postal_code
    except:
        return None, None, None


def process_bbb_data(bbb_df):
    bbb_df[['City', 'State', 'Postal_code']] = bbb_df['Address'].apply(
        lambda x: pd.Series(parse_address(x))
    )
    bbb_df['BusinessCategories'] = bbb_df['BusinessCategories'].apply(
        lambda x: x.split(',') if pd.notnull(x) else []
    )
    bbb_df['NumberOfServices'] = bbb_df['BusinessCategories'].apply(len)
    return bbb_df


pet_bbb = load_bbb_data()
pet_bbb = process_bbb_data(pet_bbb)


@st.cache_data
def load_us_states():
    return pd.read_csv('US_States.csv')


# Merge US states data (which should include center coordinates for each state) with your pet companies data
us_states = load_us_states()
pet_bbb = pd.merge(pet_bbb, us_states, on='State', how='left')

# Aggregate data by state: count companies and create a list of unique company names.
state_agg = pet_bbb.groupby('State').agg(
    count=('Company', 'nunique'),
    companies=('Company', lambda x: ", ".join(sorted(x.unique()))),
    LAT=('LAT', 'mean'),
    LON=('LON', 'mean')
).reset_index()


def usmap_distribution():
    # Create a scatter_geo plot with marker size based on company count.
    fig = px.scatter_geo(
        state_agg,
        lat="LAT",
        lon="LON",
        hover_name="State",
        hover_data={"count": True, "companies": True},
        scope="usa",
        size="count",
        size_max=50,
        title="Distribution of Pet Shipping Companies by State"
    )
    st.plotly_chart(fig)


usmap_distribution()

pet_rows = pet_bbb.explode('BusinessCategories')
pet_rows['BusinessCategories'] = pet_rows['BusinessCategories'].apply(lambda x: x.strip() if pd.notnull(x) else None)
df_top_services = pet_rows.groupby('BusinessCategories')['Company'].nunique().sort_values(
    ascending=False).reset_index().head(10)
df_top_services.columns = ['Services', 'Number_of_companies']


def top_services(df_top_services):
    st.subheader("Top Services Provided by Various Companies")
    fig = px.bar(
        df_top_services,
        x="Services", y="Number_of_companies",
        height=400,
        title='Number of Companies Providing Top 10 Services'
    )
    st.plotly_chart(fig)


top_services(df_top_services)

st.markdown("<hr>", unsafe_allow_html=True)
# ---------------------------
# Competitor Website Analysis Section
# ---------------------------
st.header("Competitor Website Analysis")
st.write(
    "Select a competitor from the list below to extract important business and SEO information, including service details, pricing (if applicable), and website stack. "
    "Alternatively, you can enter a URL manually."
)

# Merge IPATA and MapPricing data on 'Company' (adjust column names as needed).
if not IPATA_df.empty:
    ipata_pricing = pd.merge(IPATA_df, MapPricing_df, on='Company', how='left')
    # Build a mapping from a key (Company and Website) to the pricing info.
    competitor_price_mapping = {}
    ipata_options = []
    for _, row in ipata_pricing.iterrows():
        if pd.notnull(row['Website']):
            key = f"{row['Company']} ({row['Website']})"
            # Only add an indicator to the dropdown if pricing is available
            display_key = key + (" - Price Available" if pd.notnull(row.get('Pricing')) else "")
            ipata_options.append(display_key)
            competitor_price_mapping[key] = row.get('Pricing', 'NA')
    # Add an option for manual URL entry.
    ipata_options.append("Other (Enter URL manually)")
else:
    ipata_options = ["Other (Enter URL manually)"]

# Create a selectbox for competitor selection.
selected_option = st.selectbox("Select a Competitor:", options=ipata_options)

# Determine the URL based on the user's selection.
if selected_option.startswith("Other"):
    url = st.text_input("Enter Competitor's Website URL:")
    selected_company_key = None
else:
    # Remove any trailing indicator text (e.g., " - Price Available") to extract the key.
    selected_company_key = selected_option.split(" - ")[0]
    # Extract the URL from the key (it's inside parentheses).
    match = re.search(r'\((https?://[^\)]+)\)', selected_company_key)
    if match:
        url = match.group(1)
    else:
        url = None

if st.button("Gather Insights"):
    if url:
        st.info("Analyzing the website...")
        result = gather_competitor_insights(url)
        if "error" in result:
            st.error(result["error"])
        else:
            st.subheader("Website Description")
            st.write(result["description"])

            st.subheader("Services and Products Overview")
            for idx, info in enumerate(result["services_and_products"], start=1):
                st.write(f"**{idx}.** {info}")

            st.subheader("SEO Score")
            st.write(f"Estimated SEO score was calculated calculated by assigning points for the presence and quality"
                     f" of on-page elements like title tags, meta descriptions, H1 tags, image ALT attributes, and canonical links:\n **{result['seo_score']} / 100**")

            st.subheader("Years in Business")
            st.write(result["years_in_business"])

            st.subheader("Animal Services")
            animal_data = result["animal_services"]
            if isinstance(animal_data, list):
                for animal in animal_data:
                    st.write(f"- {animal.capitalize()}")
            else:
                st.write(animal_data)


            st.subheader("Pricing Information")
            # If a competitor was selected from the dropdown (not manual entry), retrieve pricing from the mapping.
            if selected_company_key and selected_company_key in competitor_price_mapping:
                pricing_value = competitor_price_mapping[selected_company_key]
                st.write(f"Estimdated quote from our database, which indicated that a 90‑lb, 30‑inch German Shepherd "
                         f"would be shipped from  NYC to London, UK via cargo"
                         f",: **{pricing_value if pd.notnull(pricing_value) else 'NA'}**")
            else:
                st.write("Pricing: **NA**")

            st.subheader("Website Stack")
            tech_stack = result["tech_stack"]
            if isinstance(tech_stack, list):
                st.write(", ".join(tech_stack))
            else:
                st.write(tech_stack)
    else:
        st.error("Please enter a valid URL.")



st.markdown("<hr>", unsafe_allow_html=True)

# ---------------------------
# Company Sentiment & Reviews Analysis Section
# ---------------------------
st.header("Sentiment & Reviews Analysis")
st.write("Select one or more companies to view sentiment analysis of Google Reviews. "
         "View sentiment analysis of Google Reviews to gain insights into customer feedback,"
         " strengths, weaknesses, review trends, and correlations between ratings and key factors"
         " like customer service and pricing.")


@st.cache_data
def load_company_data(file_path):
    df = pd.read_excel(file_path)
    strengths = df["Strengths"].apply(eval)
    weaknesses = df["Weaknesses"].apply(eval)
    return df, strengths, weaknesses


google_reviews = GoogleReview_df.copy()
#google_reviews = pd.read_excel("google_reviews.xlsx")

# Clean the "Rating" column for both cloud and local sources
google_reviews['Rating'] = pd.to_numeric(
    google_reviews['Rating']
    .str.replace(" stars", "", regex=False)
    .str.strip(),
    errors='coerce'
)

# Drop rows with missing ratings
google_reviews = google_reviews.dropna(subset=['Rating'])

# Set the column names
review_column = "Review"
company_column = "Name"

# Drop reviews missing either review text or company name
reviews = google_reviews[[review_column, company_column, "Rating"]].dropna()

# Define categories and their associated keywords.
categories = {
    "Customer Service": ["responsive", "helpful", "supportive", "service", "team"],
    "Pricing": ["affordable", "expensive", "cost", "pricing", "value"],
    "Communication": ["updates", "informative", "contact", "communicative"],
    "Ease of Process": ["easy", "smooth", "process", "hassle-free", "booking"],
    "Safety and Care": ["safe", "care", "well-being", "condition"],
    "Reputation": ["reliable", "recommend", "trustworthy", "professional", "reputation"],
}


# Function to analyze sentiment using TextBlob
def analyze_sentiment(review):
    blob = TextBlob(review)
    sentiment_score = blob.sentiment.polarity  # Polarity: -1 (negative) to +1 (positive)
    return "Positive" if sentiment_score > 0 else "Negative"


# Function to categorize a review: returns a list of categories whose keywords appear in the review
def categorize_review(review):
    review_categories = []
    for category, keywords in categories.items():
        if any(keyword in review.lower() for keyword in keywords):
            review_categories.append(category)
    return review_categories


# Apply sentiment analysis and category categorization to each review
reviews["Sentiment"] = reviews[review_column].apply(analyze_sentiment)
reviews["Categories"] = reviews[review_column].apply(categorize_review)


# Create a detailed review-level summary:
# For each review, if the sentiment is "Positive", record its categories as strengths;
# if "Negative", record as weaknesses.
def create_detailed_review_summary(df):
    df = df.copy()
    df["Detailed_Strengths"] = df.apply(lambda row: row["Categories"] if row["Sentiment"] == "Positive" else None,
                                        axis=1)
    df["Detailed_Weaknesses"] = df.apply(lambda row: row["Categories"] if row["Sentiment"] == "Negative" else None,
                                         axis=1)
    return df


detailed_review_summary = create_detailed_review_summary(reviews)

# For each company, count the number of positive and negative reviews per category.
summary = {}
for company in reviews[company_column].unique():
    company_reviews = reviews[reviews[company_column] == company]
    strengths_count = {cat: 0 for cat in categories.keys()}
    weaknesses_count = {cat: 0 for cat in categories.keys()}

    # Process each review for the current company
    for _, row in company_reviews.iterrows():
        for cat in row["Categories"]:
            if row["Sentiment"] == "Positive":
                strengths_count[cat] += 1
            elif row["Sentiment"] == "Negative":
                weaknesses_count[cat] += 1

    # Only include categories that had some reviews so the dict isn't all zeros
    summary[company] = {
        "Strengths": {cat: count for cat, count in strengths_count.items() if count > 0},
        "Weaknesses": {cat: count for cat, count in weaknesses_count.items() if count > 0},
    }

# Save company-level summary to an Excel file
output_file_path = "company_summary.xlsx"
summary_df = pd.DataFrame([
    {
        "Company": company,
        "Strengths": details["Strengths"],
        "Weaknesses": details["Weaknesses"],
    }
    for company, details in summary.items()
])
summary_df.to_excel(output_file_path, index=False)

reviews, strengths, weaknesses = load_company_data("company_summary.xlsx")


def prepare_viz_data(df, strengths, weaknesses, categories):
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
viz_df = prepare_viz_data(reviews, strengths, weaknesses, categories_list)


def plot_comparison(viz_df, selected_companies):
    filtered_df = viz_df[viz_df["Company"].isin(selected_companies)]
    strengths_fig = px.bar(
        filtered_df, x="Category", y="Strengths", color="Company",
        barmode="group", title="Strengths per Category for Selected Companies"
    )
    st.plotly_chart(strengths_fig)
    weaknesses_fig = px.bar(
        filtered_df, x="Category", y="Weaknesses", color="Company",
        barmode="group", title="Weaknesses per Category for Selected Companies"
    )
    st.plotly_chart(weaknesses_fig)


def plot_google_reviews_scatter(google_reviews, selected_companies):
    aggregated_reviews = google_reviews.groupby("Name").agg(
        Avg_Rating=("Rating", "mean"),
        Review_Count=("Rating", "size")
    ).reset_index()
    filtered_reviews = aggregated_reviews if "All" in selected_companies else aggregated_reviews[
        aggregated_reviews["Name"].isin(selected_companies)]
    scatter_fig = px.scatter(
        filtered_reviews, x="Avg_Rating", y="Review_Count", color="Name",
        title="Average Ratings vs. Rating Count for Selected Companies",
        labels={"Avg_Rating": "Average Rating (Stars)", "Review_Count": "Total Number of Ratings"},
        size_max=40
    )
    st.plotly_chart(scatter_fig)


# Sentiment and review categorization functions
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
    sentiment_score = blob.sentiment.polarity
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

    # Calculate total score by summing up all attributes
    summary_data["Total Score"] = summary_data.sum(axis=1)

    # Sort by Total Score in descending order
    summary_data = summary_data.sort_values("Total Score", ascending=False)

    return pd.DataFrame(summary_data)


all_companies = list(google_reviews["Name"].unique())
selected_companies = st.multiselect("Choose one or more companies to visualize:",
                                    options=set(all_companies),
                                    default=None)

if selected_companies:
    st.subheader("Google Reviews Analysis")
    plot_google_reviews_scatter(google_reviews, selected_companies)

    # Google Reviews Timeline Visualization
    if "Review_time" in google_reviews.columns:

        def parse_review_time(time_str):
            time_str = str(time_str).lower().strip()
            now = datetime.now()
            if time_str.startswith("a month"):
                return now - relativedelta(months=1)
            if time_str.startswith("a year"):
                return now - relativedelta(years=1)
            months_match = re.search(r"(\d+)\s+months?\s+ago", time_str)
            if months_match:
                months = int(months_match.group(1))
                return now - relativedelta(months=months)
            years_match = re.search(r"(\d+)\s+years?\s+ago", time_str)
            if years_match:
                years = int(years_match.group(1))
                return now - relativedelta(years=years)
            return now


        google_reviews["Approx_Date"] = google_reviews["Review_time"].apply(parse_review_time)
        timeline_df = google_reviews[google_reviews["Name"].isin(selected_companies)].copy()
        timeline_df = timeline_df.sort_values(by=["Name", "Approx_Date"])
        timeline_df["Cumulative_Reviews"] = timeline_df.groupby("Name").cumcount() + 1
        fig_line = px.line(
            timeline_df, x="Approx_Date", y="Cumulative_Reviews", color="Name",
            title="Cumulative Reviews Timeline by Company",
            labels={"Approx_Date": "Date", "Cumulative_Reviews": "Cumulative Reviews", "Name": "Company"}
        )
        st.plotly_chart(fig_line)
        timeline_df["Month_Year"] = timeline_df["Approx_Date"].dt.to_period("M").astype(str)
        reviews_per_month = timeline_df.groupby(["Name", "Month_Year"])["Rating"].count().reset_index()
        reviews_per_month.columns = ["Name", "Month_Year", "Reviews"]
        reviews_per_month = reviews_per_month.sort_values(by=["Name", "Month_Year"])
        fig_bar = px.bar(
            reviews_per_month, x="Month_Year", y="Reviews", color="Name",
            barmode="group",
            title="Monthly Reviews by Company",
            labels={"Month_Year": "Month-Year", "Reviews": "Number of Reviews", "Name": "Company"}
        )
        st.plotly_chart(fig_bar)
    else:
        st.info("The Google Reviews data does not contain a 'Review_time' column for timeline visualization.")

    st.subheader("Strengths & Weaknesses Comparison")
    st.text("These visualizations allow you to analyze competitors by comparing their strengths and weaknesses"
            " in key service areas like Pricing,"
            " Customer Service, and Reputation. Use them to identify market gaps, understand where competitors"
            " excel or struggle, and refine business strategy.")
    plot_comparison(viz_df, selected_companies)

    review_summary_df = create_company_review_summary(google_reviews)
    st.subheader("Positive Review Sentiment Summary for All Competitors")
    st.dataframe(review_summary_df)

    # Detailed review sentiment analysis and correlation
    detailed_review_summary = create_detailed_review_summary(google_reviews)
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
    st.write(
        "A stronger overall company strength often correlates with better ratings.\n"
        "The correlation values indicate how positive mentions in specific categories relate to ratings."
    )
    fig_corr = px.bar(
        corr_df_new, x="Category", y="Correlation",
        title="Review Rating vs. Positive Category Mentions Correlation",
        labels={"Correlation": "Correlation Value", "Category": "Review Category"},
        color="Correlation", color_continuous_scale="RdBu"
    )
    st.plotly_chart(fig_corr)

else:
    st.warning("Please select at least one company to visualize!")
