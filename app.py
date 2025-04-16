import streamlit as st
import pandas as pd
import plotly.express as px
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentiment import fetch_reddit_posts_raw, get_stock_price_data
from datetime import datetime

st.set_page_config(page_title="📈 Reddit Stock Sentiment", layout="wide")
st.title("📊 Reddit Stock Sentiment Analysis")

run_analysis = False  # control flag

# Sidebar
with st.sidebar:
    st.header("Search Settings")
    stock = st.text_input("Enter stock keyword or ticker:", value="AAPL")
    subreddit = st.selectbox("Select subreddit:", ["ALL", "wallstreetbets", "stocks", "investing"])
    limit = st.slider("Number of posts to fetch", 10, 200, 100)

    run_analysis = st.button("Run Sentiment Analysis")

# Load model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    return tokenizer, model

tokenizer, model = load_model()

# Get stock data
st.subheader("📈 Stock Price Chart")
stock_df = get_stock_price_data(stock.upper(), period="5y", interval="1d")

if stock_df.empty:
    st.warning("Could not fetch stock price data.")
else:
    stock_df["Date"] = pd.to_datetime(stock_df["Date"])
    stock_df.set_index("Date", inplace=True)
    weekly_df = stock_df["Close"].resample("W").mean().reset_index()

    current_year = datetime.now().year
    default_view_df = weekly_df[weekly_df["Date"].dt.year == current_year]

    fig = px.line(weekly_df, x="Date", y="Close", title=f"{stock.upper()} Weekly Avg Price (5y zoomable)")
    fig.update_xaxes(range=[
        default_view_df["Date"].min(),
        default_view_df["Date"].max()
    ])
    fig.update_layout(xaxis_rangeslider_visible=True)
    st.plotly_chart(fig, use_container_width=True)

# Analyze Reddit posts
if run_analysis:
    with st.spinner("Fetching Reddit posts and analyzing..."):
        posts_df = fetch_reddit_posts_raw(stock, subreddit, limit)

        if posts_df.empty:
            st.warning("No posts found.")
        else:
            sentiments, confidences = [], []
            for title in posts_df["title"]:
                inputs = tokenizer(title, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    logits = model(**inputs).logits
                    probs = F.softmax(logits, dim=1).numpy()
                    sentiments.append(np.argmax(probs))
                    confidences.append(np.max(probs))

            label_map = {0: "negative", 1: "neutral", 2: "positive"}
            posts_df["sentiment"] = [label_map[s] for s in sentiments]
            posts_df["confidence"] = [round(c * 100, 1) for c in confidences]

            st.success(f"Analyzed {len(posts_df)} posts.")

            st.subheader("Sentiment Distribution")
            dist = posts_df["sentiment"].value_counts().reset_index()
            dist.columns = ["Sentiment", "Count"]
            st.plotly_chart(px.bar(dist, x="Sentiment", y="Count", color="Sentiment"), use_container_width=True)

            st.subheader("Top 5 Posts by Sentiment (Most Confident)")
            top_5 = posts_df.sort_values(by="confidence", ascending=False).head(5)[
                ["title", "sentiment"]
            ].copy()

            sentiment_colors = {
                "positive": "green",
                "negative": "red",
                "neutral": "blue"
            }

            top_5["Sentiment"] = top_5["sentiment"].apply(
                lambda s: f"<span style='color:{sentiment_colors[s]}; font-weight:bold'>{s.capitalize()}</span>"
            )

            top_5 = top_5.rename(columns={"title": "Reddit Post"})

            st.markdown(
                top_5.to_html(escape=False, index=False),
                unsafe_allow_html=True
            )

            st.markdown("---")
            st.markdown("### 🤖 Model Used")
            st.markdown(
                """
                This app uses the **[`cardiffnlp/twitter-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)** model  
                from Hugging Face. It is trained on millions of tweets and classifies text into **Positive**, **Neutral**, or **Negative**.  

                Its understanding of informal social media language makes it ideal for analyzing Reddit posts related to stock sentiment.
                """
            )
