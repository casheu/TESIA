import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go

def run():
    st.markdown('---')
    
    # Title
    st.subheader("Price Prediction")

    st.markdown('---')

    stock = st.selectbox('Pick a stock:', ('BBNI', 'BBRI', 'BBTN', 'BMRI'))

    # Import scaler and model
    with open(stock + '_scaler.pkl', 'rb') as file_1:
      scaler = joblib.load(file_1)

    model = tf.keras.models.load_model(stock + '_Mod')

    # Scrapping
    stock = yf.Ticker(stock + ".JK")
    hist_all = stock.history(period="max").reset_index()
    hist_lat = stock.history(period="min").reset_index()
    year_rn = hist_lat['Date'].iloc[0].year
    month_rn = hist_lat['Date'].iloc[0].month
    SY_SM = hist_all[(hist_all['Date'].dt.year == year_rn) & (hist_all['Date'].dt.month == month_rn)]
    
    st.markdown('---')

    st.subheader("Price History")

    fig = plt.figure(figsize=(15, 5))
    plt.plot(SY_SM['Date'], SY_SM['Close'], label = 'Actual')
    plt.legend()
    st.pyplot(fig)

    st.markdown('---')

    if st.button('Predict Price'):
    
      last_15 = hist_all[['Close']].tail(15).reset_index(inplace = False, drop = True)

      # Scaled and transposed last 15 close price
      last_15_scaled = scaler.transform(last_15)
      last_15_T = last_15_scaled.T

      # Predict h+1
      Predict_h1 = model.predict(last_15_T)
      Predict_true = scaler.inverse_transform(Predict_h1)
      Predict_true = pd.DataFrame(Predict_true)

      dateall = pd.DataFrame(pd.DatetimeIndex(hist_all['Date']) + pd.DateOffset(1))
      last_day = dateall[['Date']]
      Predict_true['Date'] = last_day.tail(1).reset_index(inplace=False,drop=True)

      hist3m = hist_all.tail(75)

      st.markdown('---')
      
      st.subheader("Recent Prices and Prediction")

      fig = go.Figure()
      fig.add_trace(go.Candlestick(x=hist3m['Date'],
                      open=hist3m['Open'],
                      high=hist3m['High'],
                      low=hist3m['Low'],
                      close=hist3m['Close']))
      fig.add_trace(go.Scatter(x=Predict_true['Date'], y=Predict_true[0]))
      st.plotly_chart(fig)
      
      st.markdown('---')
      
      st.write('## Prediction : ', Predict_true.at[0,0])
      
      st.markdown('---')

if __name__ == '__main__':
    run()