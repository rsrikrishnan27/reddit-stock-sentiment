# --- sentiment.py ---

import praw
import yfinance as yf
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

# Load Reddit API credentials from secrets or environment
def get_reddit_instance():
    return praw.Reddit(
        client_id=os.getenv("REDDIT_CLIENT_ID"),
        client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
        user_agent=os.getenv("REDDIT_USER_AGENT"),
    )

def fetch_reddit_posts_raw(stock_keyword, subreddit_choice, limit=100):
    reddit = get_reddit_instance()

    subreddits = (
        ["wallstreetbets", "stocks", "investing"]
        if subreddit_choice == "ALL"
        else [subreddit_choice]
    )

    posts = []
    for sub in subreddits:
        for post in reddit.subreddit(sub).search(stock_keyword, sort="new", limit=limit // len(subreddits)):
            posts.append({
                "title": post.title,
                "subreddit": sub,
                "created_utc": pd.to_datetime(post.created_utc, unit="s"),
                "score": post.score
            })

    return pd.DataFrame(posts)


def get_stock_price_data(ticker, period="5y", interval="1d"):
    try:
        df = yf.download(ticker, period=period, interval=interval)
        df = df.reset_index()
        return df
    except Exception as e:
        print(f"Error fetching stock data: {e}")
        return pd.DataFrame()
