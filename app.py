### --- app.py ---

import streamlit as st
import plotly.express as px
from sentiment import fetch_reddit_posts, get_stock_price_data
import pandas as pd
import threading
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import torch.nn.functional as F

st.set_page_config(page_title="📈 Reddit Stock Sentiment", layout="wide")
st.title("📊 Reddit Stock Sentiment Analysis")

# --- Sidebar input ---
with st.sidebar:
    st.header("🔍 Search Settings")
    stock = st.text_input("Enter a stock keyword or ticker:", value="AAPL")
    subreddit = st.selectbox("Select subreddit:", ["ALL", "wallstreetbets", "stocks", "investing"])
    limit = st.slider("Number of posts to fetch:", min_value=10, max_value=200, value=100)

    if "run_sentiment" not in st.session_state:
        st.session_state.run_sentiment = False
    if "sentiment_loading" not in st.session_state:
        st.session_state.sentiment_loading = False
    if "sentiment_df" not in st.session_state:
        st.session_state.sentiment_df = {}

    if st.button("Run Sentiment Analysis") and not st.session_state.sentiment_loading:
        st.session_state.run_sentiment = True
        st.session_state.sentiment_loading = True

        def run_sentiment_thread():
            from reddit_sentiment import fetch_reddit_posts_raw
            df_raw = fetch_reddit_posts_raw(stock, subreddit, limit)

            @st.cache_resource
            def load_model():
                tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
                model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
                return tokenizer, model

            tokenizer, model = load_model()

            texts = df_raw['title'].tolist()
            batch_size = 32
            sentiments, confidences = [], []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = F.softmax(outputs.logits, dim=1).numpy()
                    labels = np.argmax(probs, axis=1)
                    sentiments.extend(labels)
                    confidences.extend(np.max(probs, axis=1))

            label_map = {0: "negative", 1: "neutral", 2: "positive"}
            df_raw['sentiment'] = [label_map[l] for l in sentiments]
            df_raw['confidence'] = confidences

            st.session_state.sentiment_df[stock.upper()] = df_raw
            st.session_state.sentiment_loading = False

        threading.Thread(target=run_sentiment_thread).start()

# --- Stock Chart Section ---
st.subheader("📈 Stock Price Trend")

if "interval_option" not in st.session_state:
    st.session_state.interval_option = "D"

col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("D"):
        st.session_state.interval_option = "D"
with col2:
    if st.button("W"):
        st.session_state.interval_option = "W"
with col3:
    if st.button("M"):
        st.session_state.interval_option = "M"
with col4:
    if st.button("Y"):
        st.session_state.interval_option = "Y"

interval_option = st.session_state.interval_option
period, interval = "5y", "1d"
if interval_option == "W":
    interval = "1wk"
elif interval_option == "M":
    interval = "1mo"
elif interval_option == "Y":
    interval = "3mo"

stock_price_df = get_stock_price_data(stock.upper(), period=period, interval=interval)

if stock_price_df.empty:
    st.warning("📉 Could not fetch stock data. Check ticker symbol.")
else:
    fig = px.line(
        stock_price_df,
        x="Date",
        y="Close",
        title=f"{stock.upper()} Stock Price - Interval: {interval_option}",
        labels={"Close": "Price (USD)", "Date": "Date"},
    )

    latest_date = stock_price_df["Date"].dt.tz_localize(None).max()
    default_start = latest_date - pd.Timedelta(days=30)

    fig.update_layout(
        autosize=True,
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider=dict(visible=True),
        xaxis=dict(range=[default_start, latest_date])
    )

    st.plotly_chart(fig, use_container_width=True)

# --- Sentiment analysis section ---
if st.session_state.sentiment_loading:
    st.info("⌛ Analyzing Reddit posts...")
elif stock.upper() in st.session_state.sentiment_df:
    df = st.session_state.sentiment_df[stock.upper()]
    st.success(f"✅ Analyzed {len(df)} posts mentioning '{stock}'.")

    st.subheader("📊 Sentiment Distribution")
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    fig = px.bar(
        sentiment_counts,
        x='Count',
        y='Sentiment',
        orientation='h',
        color='Sentiment',
        title='Distribution of Sentiment (Positive / Neutral / Negative)'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🏅 Top 5 Reddit Posts by Sentiment Confidence")
    top_5 = df.sort_values(by="confidence", ascending=False).head(5)

    sentiment_colors = {
        "positive": "#2ecc71",
        "negative": "#e74c3c",
        "neutral": "#3498db"
    }

    for _, row in top_5.iterrows():
        sentiment = row['sentiment']
        color = sentiment_colors.get(sentiment, "#bdc3c7")
        confidence = round(row['confidence'] * 100, 1)

        st.markdown(f"**{row['title']}**")
        st.markdown(
            f"""
            <span style="font-size: 14px;">
            <b>Sentiment:</b> <span style="background-color:{color}; padding:3px 8px; border-radius:4px; color:white;">
            {sentiment.capitalize()}</span>
            &nbsp; | &nbsp;
            <b>Confidence:</b> {confidence}% &nbsp; | &nbsp;
            <b>Subreddit:</b> r/{row['subreddit']}
            </span>
            """,
            unsafe_allow_html=True
        )
        st.markdown("---")

    st.subheader("🧾 Reddit Posts with Sentiment + Confidence")
    df_display = df[["title", "subreddit", "confidence", "sentiment"]].copy()
    df_display["Confidence %"] = (df_display["confidence"] * 100).round(1)
    df_display = df_display.drop(columns=["confidence"])

    st.dataframe(
        df_display.sort_values(by="Confidence %", ascending=False).reset_index(drop=True),
        use_container_width=True,
        column_config={
            "Confidence %": st.column_config.ProgressColumn(
                "Confidence",
                help="Prediction confidence from the model",
                format="%.1f",
                min_value=0.0,
                max_value=100.0
            )
        }
    )

    st.markdown("---")
    st.markdown("### 🤖 Model Used")
    st.markdown(
        """
        This app uses the **[`cardiffnlp/twitter-roberta-base-sentiment`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)** model 
        from HuggingFace. It is trained on millions of tweets and classifies text into **Positive**, **Neutral**, or **Negative**. 

        Its understanding of social media language and informal expressions makes it ideal for analyzing Reddit posts related to stock sentiment.
        """
    )
