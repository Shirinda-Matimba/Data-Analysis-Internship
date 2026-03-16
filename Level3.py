import pandas as pd
import nltk
import re
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from textblob import TextBlob
from wordcloud import WordCloud

# Tast 3: Natural Language Processing(NLP)-Sentiment Analysis
# Download NLTK resources only needed first time
nltk.download('punkt')
nltk.download('stopwords')

# Load the sentiment dataset
df = pd.read_csv("sentiment.csv")
print(df.columns)

print("First 5 rows:")
print(df.head())

# Check dataset information
print("\nDataset info:")
print(df.info())

# Text Cleaning Function

# The function below will result:
# Convert text to lowercase
# Remove punctuation
# Remove stopwords
# Apply stemming

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize words
    words = word_tokenize(text)
    
    # Remove stopwords
    words = [word for word in words if word not in stop_words]
    
    # Apply stemming
    words = [stemmer.stem(word) for word in words]
    
    # Join words back
    return " ".join(words)

# Apply text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text
df["clean_review"] = df["Text"].apply(clean_text)

print("\nCleaned text sample:")
print(df[["Text", "clean_review"]].head())

# Sentiment Analysis Function
def get_sentiment(text):
    
    analysis = TextBlob(text)
    
    if analysis.sentiment.polarity > 0:
        return "Positive"
    
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    
    else:
        return "Neutral"

# Apply Sentiment Analysis
df["Sentiment"] = df["clean_review"].apply(get_sentiment)

print("\nSentiment results:")
print(df[["clean_review", "Sentiment"]].head())

# Sentiment Distribution
sentiment_counts = df["Sentiment"].value_counts()

print("\nSentiment Distribution:")
print(sentiment_counts)

# Plot sentiment distribution
sentiment_counts.plot(kind="bar")

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Reviews")
plt.show()

# Generate Word Cloud
# Combine all text
text = " ".join(df["clean_review"])
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud)
plt.axis("off")
plt.title("Word Cloud of Reviews")
plt.show()