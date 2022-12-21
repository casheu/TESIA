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
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re
import numpy as np
import seaborn as sns

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
            st.markdown('---')
        with col2:
            st.subheader("Price Prediction")
            st.markdown('---')

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Today's Tweet")
            st.markdown('---')

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

            st.dataframe(tweets)
            st.markdown('---')

        with col2:
            st.subheader("Price History")
            st.markdown('---')

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
            st.markdown('---')

    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            if st.button('Predict sentiment'):
                st.markdown('---')
                df = pd.DataFrame()
        
                # Mendefinisikan stopwords bahasa Indonesia
                idn = list(set(stopwords.words('indonesian')))
                factory = StemmerFactory()
                stemmer = factory.create_stemmer()

                def text_process(text):
                    # Mengubah Teks ke Lowercase
                    text = text.lower()
                    
                    # Menghilangkan Mention
                    text = re.sub("@[A-Za-z0-9_]+", " ", text)
                    
                    # Menghilangkan Hashtag
                    text = re.sub("#[A-Za-z0-9_]+", " ", text)
                    
                    # Menghilangkan \n
                    text = re.sub(r"\\n", " ",text)
                    
                    # Menghilangkan Whitespace
                    text = text.strip()

                    # Menghilangkan Link
                    text = re.sub(r"http\S+", " ", text)
                    text = re.sub(r"www.\S+", " ", text)

                    # Menghilangkan yang Bukan Huruf seperti Emoji, Simbol Matematika (seperti μ), dst
                    text = re.sub("[^A-Za-z\s']", " ", text)

                    # Menghilangkan RT
                    text = re.sub("rt", " ",text)

                    # Melakukan Tokenisasi
                    tokens = word_tokenize(text)

                    # Menghilangkan Stopwords
                    text = ' '.join([word for word in tokens if word not in idn])

                    # Melakukan Stemming
                    text = ' '.join([stemmer.stem(word) for word in text.split()])
                    
                    return text

                df['Tweet_processed'] = tweets['Tweet'].apply(lambda x: text_process(x))
                pred = np.argmax(nlp.predict(df['Tweet_processed']), axis=-1)
                pred_df = pd.DataFrame(pred, columns=['label'])
                pred_df['label'] = pred_df.replace([0,1,2], ['positive','neutral','negative'])

                def PieComposition(dataframe, column):
                    palette_color = sns.color_palette('pastel')
                    data = {}
                    freq = {}
                    datalen = len(dataframe[column].unique())
                    x = np.arange(datalen)
                    dq = dataframe[column].unique()
                    for i in x:
                        data[i] = dq[i]
                        freq[i] = dataframe[column][dataframe[column] == dq[i]].value_counts().sum()
                    data = list(data.values())
                    freq = list(freq.values())
                    fig = plt.figure(figsize=(10, 5))
                    plt.pie(freq, labels = data, colors=palette_color, autopct='%.0f%%')
                    plt.show()
                    st.pyplot(fig)

                st.subheader("Sentiment Percentage")
                PieComposition(pred_df, 'label')
                st.markdown('---')

        with col2:
            st.subheader("Price Prediction")
            st.markdown('---')

if __name__ == '__main__':
    run()