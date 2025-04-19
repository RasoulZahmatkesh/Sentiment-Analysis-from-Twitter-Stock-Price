import snscrape.modules.twitter as sntwitter
import pandas as pd
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import yfinance as yf
import matplotlib.pyplot as plt
import torch

# ----------------------
# Configuration
# ----------------------
TICKER = "AAPL"
QUERY = f"{TICKER} OR ${TICKER}"
LIMIT = 300
DAYS = 7
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

# ----------------------
# 1. Fetch Tweets
# ----------------------
def get_tweets(query, days=7, limit=300):
    since_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    tweets = []
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{query} since:{since_date}').get_items()):
        if i >= limit:
            break
        tweets.append([tweet.date, tweet.content])
    df = pd.DataFrame(tweets, columns=['datetime', 'text'])
    df['date'] = pd.to_datetime(df['datetime']).dt.date
    return df

# ----------------------
# 2. Load Sentiment Model (RoBERTa)
# ----------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# ----------------------
# 3. Run Sentiment Analysis
# ----------------------
def classify_sentiment(text):
    try:
        result = sentiment_pipeline(text[:512])[0]
        return result['label']
    except:
        return "unknown"

# ----------------------
# 4. Apply Sentiment
# ----------------------
df_tweets = get_tweets(QUERY, DAYS, LIMIT)
df_tweets['sentiment'] = df_tweets['text'].apply(classify_sentiment)

# ----------------------
# 5. Get Stock Prices
# ----------------------
stock = yf.Ticker(TICKER)
df_price = stock.history(period=f"{DAYS}d")[['Close']]
df_price.reset_index(inplace=True)
df_price['Date'] = df_price['Date'].dt.date

# ----------------------
# 6. Aggregate Sentiments
# ----------------------
daily_sentiment = df_tweets.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
df_merged = df_price.merge(daily_sentiment, left_on="Date", right_on="date", how="left").fillna(0)

# ----------------------
# 7. Save and Plot
# ----------------------
df_merged.to_csv("sentiment_vs_price.csv", index=False)

plt.figure(figsize=(12, 6))
plt.plot(df_merged["Date"], df_merged["Close"], label="Stock Price", color='blue', marker='o')

if 'LABEL_2' in df_merged.columns:
    plt.bar(df_merged["Date"], df_merged["LABEL_2"], width=0.3, alpha=0.5, color='green', label="Positive")
if 'LABEL_0' in df_merged.columns:
    plt.bar(df_merged["Date"], -df_merged["LABEL_0"], width=0.3, alpha=0.5, color='red', label="Negative")

plt.title(f"Twitter Sentiment vs {TICKER} Stock Price")
plt.xlabel("Date")
plt.ylabel("Price / Tweet Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
