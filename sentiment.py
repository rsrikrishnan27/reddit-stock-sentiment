### --- reddit_sentiment.py ---

import os
import streamlit as st
import praw
import datetime
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import re
import emoji
import yfinance as yf

# --- Reddit API setup ---
client_id = st.secrets["CLIENT_ID"]
client_secret = st.secrets["CLIENT_SECRET"]
user_agent = st.secrets["USER_AGENT"]

reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent
)

# --- HuggingFace model setup ---
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
labels = ['negative', 'neutral', 'positive']

# --- Preprocess text ---
def preprocess(text):
    text = emoji.demojize(text)
    text = re.sub(r"http\S+|@\S+|#\S+|[^A-Za-z0-9\s]", "", text)
    return text.strip()

# --- Classify sentiment ---
def classify_sentiment(text):
    try:
        text = preprocess(text)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        label_index = probs.argmax().item()
        label = labels[label_index]
        confidence = round(probs[0][label_index].item(), 3)
        return label, confidence
    except Exception as e:
        print("Error during sentiment classification:", e)
        return "neutral", 0.0

# --- Fetch Reddit Posts ---
def fetch_reddit_posts(stock, subreddit='wallstreetbets', limit=100):
    subreddits = ['wallstreetbets', 'stocks', 'investing'] if subreddit == 'ALL' else [subreddit]
    all_posts = []

    for sub in subreddits:
        posts = reddit.subreddit(sub).search(stock, sort='new', limit=limit // len(subreddits))
        for post in posts:
            content = post.title + " " + (post.selftext or "")
            sentiment, confidence = classify_sentiment(content)
            all_posts.append({
                'title': post.title,
                'text': post.selftext,
                'created': datetime.datetime.fromtimestamp(post.created),
                'score': post.score,
                'sentiment': sentiment,
                'confidence': confidence,
                'subreddit': sub
            })

    return pd.DataFrame(all_posts)

# --- Fetch Stock Price Data ---
def get_stock_price_data(ticker, period="1mo", interval="1d"):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        df.reset_index(inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return pd.DataFrame()