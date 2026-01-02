# Multi-Source Sentiment Analysis for Stocks & Crypto

This project performs sentiment analysis on tweets related to a specific stock (e.g., $AAPL) using a transformer-based model (RoBERTa), then compares it with the stock's price over the same period.

# 🧰 Tools
- Python
- snscrape (Multi-Source)
- HuggingFace Transformers
- RoBERTa Model: `cardiffnlp/twitter-roberta-base-sentiment`
- yfinance (Stock data)
- matplotlib & pandas

# 🚀 Run

```bash
pip install -r requirements.txt
python sentiment_analysis.py
```

# 📈 Output

- `sentiment_vs_price.csv`: merged data
- Chart showing stock price vs sentiment counts

# 📌 Notes

- You can change the stock symbol in the script.
- Adjust `DAYS` and `LIMIT` to fetch more or fewer tweets.
