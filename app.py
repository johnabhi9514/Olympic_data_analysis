import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from tensorflow.keras.models import load_model
import streamlit as st

start = '2010-01-01'
end   = '2022-03-10'

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')
df = data.DataReader(user_input, 'yahoo', start, end)

## Describe Data
st.subheader('Data from 2010 - 2022')
st.write(df.describe())


##Visulization
st.subheader('Closing Price Vs Time Chat')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, label='Closing Price')
plt.legend()
st.pyplot(fig)


st.subheader('Closing Price Vs Time Chat with 100MA')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, label='Closing Price')
plt.plot(ma100, label='100 Moving Average')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price Vs Time Chat with 200MA')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, label='Closing Price')
plt.plot(ma100, label='200 Moving Average')
plt.plot(ma200, label='200 Moving Average')
plt.legend()
st.pyplot(fig)

## Splitting the dataset
data_train = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_test = pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

## Scalling dataset
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_train_array = scaler.fit_transform(data_train)
data_test_array = scaler.fit_transform(data_test)
  

## load Model

model = load_model('keras_model.h5')

## Testing

past_100_days = data_train.tail(100)
final_df_test = past_100_days.append(data_test, ignore_index=True)
input_test_data = scaler.fit_transform(final_df_test)

x_test = []
y_test = []

for i in range(100, input_test_data.shape[0]):
    x_test.append(input_test_data[i-100:i])
    y_test.append(input_test_data[i,0])

x_test, y_test = np.array(x_test),np.array(y_test)

## make prediction

y_pred = model.predict(x_test)

scale = scaler.scale_
scale_factor = 1/scale[0]

y_pred = y_pred * scale_factor
y_test = y_test * scale_factor


## Visualize Pred
st.subheader('Original Price Vs Predicted Price')
fig2=plt.figure(figsize=(15,8))
plt.plot(y_test, 'b', label='Original CLose Price')
plt.plot(y_pred, 'r', label='Predicted CLose Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

