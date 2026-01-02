# =========================================================
# Multi-Source Sentiment Analysis for Stocks & Crypto
# =========================================================

# ----------------------
# API CONFIGURATION
# ----------------------
REDDIT_CLIENT_ID = "YOUR_REDDIT_CLIENT_ID"
REDDIT_CLIENT_SECRET = "YOUR_REDDIT_CLIENT_SECRET"
REDDIT_USER_AGENT = "sentiment_app"

NEWS_API_KEY = "YOUR_NEWSAPI_KEY"

# ----------------------
# ASSET CONFIGURATION
# ----------------------
ASSET_NAME = "Bitcoin"
SYMBOL = "BTC"
MARKET_TYPE = "crypto"   # "stock" or "crypto"
DAYS = 7
LIMIT = 300

# =========================================================
# IMPORTS
# =========================================================
import snscrape.modules.twitter as sntwitter
import praw
from newsapi import NewsApiClient
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# =========================================================
# 1. DATA SOURCES
# =========================================================

def fetch_twitter(query, days, limit):
    since = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    texts = []

    scraper = sntwitter.TwitterSearchScraper(
        f"{query} lang:en since:{since} -filter:retweets"
    )

    for i, tweet in enumerate(scraper.get_items()):
        if i >= limit:
            break
        texts.append(tweet.content)

    return texts


def fetch_reddit(query, limit):
    reddit = praw.Reddit(
        client_id=REDDIT_CLIENT_ID,
        client_secret=REDDIT_CLIENT_SECRET,
        user_agent=REDDIT_USER_AGENT
    )

    texts = []
    for post in reddit.subreddit("all").search(query, limit=limit):
        texts.append(post.title + " " + post.selftext)

    return texts


def fetch_news(query, days):
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    from_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

    articles = newsapi.get_everything(
        q=query,
        from_param=from_date,
        language="en",
        sort_by="relevancy"
    )

    return [
        a["title"] + " " + (a["description"] or "")
        for a in articles["articles"]
    ]

# =========================================================
# 2. SENTIMENT MODEL
# =========================================================

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    truncation=True,
    max_length=512
)

def analyze_sentiment(texts):
    counts = {"LABEL_0": 0, "LABEL_1": 0, "LABEL_2": 0}

    for text in texts:
        try:
            label = sentiment_pipeline(text)[0]["label"]
            counts[label] += 1
        except:
            pass

    score = counts["LABEL_2"] - counts["LABEL_0"]
    return counts, score

# =========================================================
# 3. PRICE FETCH
# =========================================================

def get_price(symbol, market_type, days):
    if market_type == "stock":
        ticker = symbol
    else:
        ticker = f"{symbol}-USD"

    data = yf.Ticker(ticker).history(period=f"{days}d")
    return data["Close"].iloc[-1]

# =========================================================
# 4. INSIGHT ENGINE (TEXT ANALYSIS)
# =========================================================

def generate_insight(asset, symbol, price, counts, score):
    if score > 20:
        mood = "Strongly Bullish"
    elif score > 5:
        mood = "Mildly Bullish"
    elif score < -20:
        mood = "Strongly Bearish"
    elif score < -5:
        mood = "Mildly Bearish"
    else:
        mood = "Neutral"

    return f"""
Asset: {asset} ({symbol})
Market Price: {price:.2f}

Sentiment Summary:
- Positive: {counts['LABEL_2']}
- Neutral: {counts['LABEL_1']}
- Negative: {counts['LABEL_0']}

Overall Market Mood: {mood}

Interpretation:
Public discourse across social media and news sources suggests a {mood.lower()}
sentiment toward {asset}. This reflects the prevailing psychological bias of
market participants over the last {DAYS} days.
"""

# =========================================================
# 5. PIPELINE EXECUTION
# =========================================================

query = f"{ASSET_NAME} OR {SYMBOL}"

twitter_data = fetch_twitter(query, DAYS, LIMIT)
reddit_data = fetch_reddit(query, LIMIT // 3)
news_data = fetch_news(query, DAYS)

all_texts = twitter_data + reddit_data + news_data

sentiment_counts, sentiment_score = analyze_sentiment(all_texts)

price = get_price(SYMBOL, MARKET_TYPE, DAYS)

report = generate_insight(
    ASSET_NAME,
    SYMBOL,
    price,
    sentiment_counts,
    sentiment_score
)

print(report)
