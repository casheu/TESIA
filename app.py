import streamlit as st
import twitter
import prediction

# Title
st.title('TESIA')

navigation = st.sidebar.selectbox('Pages : ', ('Twitter Sentiment', 'Price Prediction'))

if navigation == 'Twitter Sentiment':
    twitter.run()
else:
    prediction.run()