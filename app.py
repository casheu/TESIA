import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow as tf
import datetime
import snscrape.modules.twitter as sntwitter
import pandas as pd
import joblib
import yfinance as yf
import matplotlib.pyplot as plt

st.set_page_config(
    page_title='TESIA',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.markdown('---')

# Download
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Import model
nlp = tf.keras.models.load_model('model_nlp')

def run():
    stock = st.selectbox('Pick a stock:', ('BBNI', 'BBRI', 'BBTN', 'BMRI'))

    st.markdown('---')

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Twitter Sentiment')
        with col2:
            st.subheader("Price Prediction")

    with st.container():
        col1, col2 = st.columns(2)
        with col2:
            st.markdown('---')

            st.subheader("Today's Tweet")

            attributes_container = []
            today = datetime.datetime.now()
            today = today.strftime('%Y-%m-%d')
            yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
            yesterday = yesterday.strftime('%Y-%m-%d')

            # Using TwitterSearchScraper to scrape data and append tweets to list
            for i,tweet in enumerate(sntwitter.TwitterSearchScraper(stock.lower() + ' since:' + yesterday + ' until:' + today).get_items()):
                attributes_container.append([tweet.user.username, tweet.date, tweet.likeCount, tweet.sourceLabel, tweet.content])

            # Creating a dataframe to load the list
            tweets_df = pd.DataFrame(attributes_container, columns=["User", "Date Created", "Number of Likes", "Source of Tweet", "Tweet"])
            tweets = tweets_df[tweets_df['Tweet'].str.contains(stock)]
            tweets_daily = pd.DataFrame(pd.to_datetime(tweets['Date Created']).dt.tz_localize(None))
            tweets['Date Created'] = tweets_daily
            tweets['Date Created'] = pd.to_datetime(tweets['Date Created']).dt.date

            st.markdown('---')

            st.dataframe(tweets)

        with col1:
            st.markdown('---')

            st.subheader("Price History")

            # Scrapping
            stock = yf.Ticker(stock + ".JK")
            hist_all = stock.history(period="max").reset_index()
            hist_lat = stock.history(period="min").reset_index()
            year_rn = hist_lat['Date'].iloc[0].year
            month_rn = hist_lat['Date'].iloc[0].month
            SY_SM = hist_all[(hist_all['Date'].dt.year == year_rn) & (hist_all['Date'].dt.month == month_rn)]

            fig = plt.figure(figsize=(15, 5))
            plt.plot(SY_SM['Date'], SY_SM['Close'], label = 'Actual')
            plt.legend()
            st.pyplot(fig)

if __name__ == '__main__':
    run()