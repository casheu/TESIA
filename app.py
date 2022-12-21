import streamlit as st
import sentiment
import prediction

# Title
st.title('TESIA')

navigation = st.sidebar.selectbox('Pages : ', ('Twitter Sentiment', 'Price Prediction'))

selectstock = st.sidebar.image('Pick a stock:', ('BBNI', 'BBRI', 'BBTN', 'BMRI'))

if navigation == 'Twitter Sentiment':
    stock = selectstock
    sentiment.run()
else:
    stock = selectstock
    prediction.run()