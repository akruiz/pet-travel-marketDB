import pandas as pd
from textblob import TextBlob
from app import GoogleReview_df

# ----------------------------
# 1. Data Loading and Preprocessing
# ----------------------------

# Load the dataset
#file_path = "google.xlsx"  # Update this with your file path
#df = pd.read_excel(file_path)
df = GoogleReview_df

# Set the column names
review_column = "Review"
company_column = "Name"

# Drop reviews missing either review text or company name
reviews = df[[review_column, company_column, "Rating"]].dropna()

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
