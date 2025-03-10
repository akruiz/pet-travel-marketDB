import pandas as pd
import matplotlib.pyplot as plt
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

# Import data
df_excel = pd.read_excel("Google Maps Reviews-Pet Transport.xlsx", engine="openpyxl")
df = df_excel[['Name', 'Category', 'Rating', 'Rating_count', 'Reviewer', 'Review_time', 'Review']]
df['Review'] = df['Review'].fillna('')


# Sentiment Analysis VADER - https://www.geeksforgeeks.org/python-sentiment-analysis-using-vader/
sia = SentimentIntensityAnalyzer()

def get_sentiment_score(review):
    if isinstance(review, str):
        sentiment = sia.polarity_scores(review)
        return sentiment['compound']
    return 0.0

df['Sentiment_Score'] = df['Review'].apply(get_sentiment_score)


def classify_sentiment(score):
    if score > 0.05:
        return 'Positive'
    elif score < -0.05:
        return 'Negative'
    else:
        return 'Neutral'

df.loc[:, 'Sentiment'] = df['Sentiment_Score'].apply(classify_sentiment)
# print(df[['Name', 'Review', 'Sentiment_Score', 'Sentiment']])


# Top 10 common words
def get_common_words(group):
    # Remove empty
    group = group[group['Review'].str.strip() != '']
    if group.empty:
        return pd.DataFrame(columns=['Word', 'Frequency'])

    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(group['Review'])
    word_freq = X.sum(axis=0).A1
    words = vectorizer.get_feature_names_out()

    word_freq_df = pd.DataFrame(list(zip(words, word_freq)), columns=['Word', 'Frequency'])
    word_freq_df = word_freq_df.sort_values(by='Frequency', ascending=False)

    # Top 10 most common words,  adjust the numbers here
    return word_freq_df.head(10)

common_words_dict = {}

for business, group in df.groupby('Name'):
    common_words_df = get_common_words(group)
    common_words = ', '.join(common_words_df['Word'])
    common_words_dict[business] = common_words

df['Common_Words'] = df['Name'].map(common_words_dict)
# print(df[['Name', 'Common_Words']])


columns_export = ['Name', 'Category', 'Rating', 'Rating_count', 'Reviewer', 'Review_time', 'Review', 'Sentiment_Score', 'Sentiment', 'Common_Words']
df[columns_export].to_csv("Google Review Sentiment Analysis.csv", index=False)