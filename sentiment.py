import os
import praw
import datetime
import pandas as pd
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import re
import emoji
import asyncio
import sys

if sys.platform.startswith('win') and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    
# Load secrets
load_dotenv("secrets.txt")

# Reddit API setup
reddit = praw.Reddit(
    client_id=os.getenv("CLIENT_ID"),
    client_secret=os.getenv("CLIENT_SECRET"),
    user_agent=os.getenv("USER_AGENT")
)

# Load model + tokenizer
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
labels = ['negative', 'neutral', 'positive']

def preprocess(text):
    text = emoji.demojize(text)
    text = re.sub(r"http\S+|@\S+|#\S+|[^A-Za-z0-9\s]", "", text)
    return text.strip()

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
