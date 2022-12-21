import streamlit as st
import nltk
import tensorflow as tf

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
    with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Twitter Sentiment')
    with col2:
        st.subheader("Price Prediction")

if __name__ == '__main__':
    run()