import streamlit as st
import sentiment
import prediction

# Title
st.title('TESIA')

navigation = st.sidebar.selectbox('Pages : ', ('Twitter Sentiment', 'Price Prediction'))

selectstock = st.sidebar.selectbox('Pick a stock:', ('BBNI', 'BBRI', 'BBTN', 'BMRI'))

if navigation == 'Twitter Sentiment':
    sentiment.run()
else:
    prediction.run()