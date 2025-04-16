### --- app.py ---

import streamlit as st
import plotly.express as px
from sentiment import fetch_reddit_posts, get_stock_price_data
import pandas as pd

st.set_page_config(page_title="📈 Reddit Stock Sentiment", layout="wide")
st.title("📊 Reddit Stock Sentiment Analysis")

# Sidebar input
with st.sidebar:
    st.header("🔍 Search Settings")
    stock = st.text_input("Enter a stock keyword or ticker:", value="AAPL")
    subreddit = st.selectbox("Select subreddit:", ["ALL", "wallstreetbets", "stocks", "investing"])
    limit = st.slider("Number of posts to fetch:", min_value=10, max_value=200, value=100)
    analyze_btn = st.button("Run Sentiment Analysis")

# --- Always show stock price chart first ---
st.subheader("📈 Stock Price Trend")
interval_option = st.radio("Select interval:", ["D", "W", "M", "Y"], horizontal=True)

# Determine period and interval
if interval_option == "D":
    period, interval = "30d", "1d"
elif interval_option == "W":
    period, interval = "ytd", "1wk"
elif interval_option == "M":
    period, interval = "1y", "1mo"
else:  # "Y"
    period, interval = "5y", "3mo"

stock_price_df = get_stock_price_data(stock, period=period, interval=interval)

if stock_price_df.empty:
    st.warning("📉 Could not fetch stock data. Check ticker symbol.")
else:
    fig = px.line(
        stock_price_df,
        x="Datetime",
        y="Close",
        title=f"{stock.upper()} Stock Price - Interval: {interval_option}",
        labels={"Close": "Price (USD)", "Datetime": "Date"}
    )
    fig.update_layout(
        autosize=True,
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis_title="Date",
        yaxis_title="Price (USD)"
    )
    st.plotly_chart(fig, use_container_width=True)

# --- Sentiment analysis section ---
if analyze_btn:
    with st.spinner("Fetching and analyzing Reddit posts..."):
        df = fetch_reddit_posts(stock, subreddit, limit)

    if df.empty:
        st.warning("⚠️ No Reddit posts found. Try another keyword.")
    else:
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
