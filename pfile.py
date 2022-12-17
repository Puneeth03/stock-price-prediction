import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense, Conv1D
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('C:/Users/sulakshan/Desktop/dm_project/ADANIPORTS.csv' )
data.head()

data.dropna(axis = 0, inplace = True)

data = data.drop(columns=['Symbol', 'Series', 'Prev Close','Last', 'VWAP', 'Turnover', 'Trades', 'Deliverable Volume', '%Deliverble'])
data.shape

data['dma'] = data['Close'] - data['Close'].shift(5)
data['dma_positive'] = np.where(data['dma'] > 0, 1, 0)

data.set_index('Date', inplace = True)
data.head()

scaler = MinMaxScaler(feature_range=(0,1))

X = data[['Open', 'Low', 'High', 'Volume', 'dma_positive']].copy()
y = data['Close'].copy()

X[['Open', 'Low', 'High', 'Volume', 'dma_positive']] = scaler.fit_transform(X)
y = scaler.fit_transform(y.values.reshape(-1, 1))

def load_data(X, seq_len, train_size=0.8):
    amount_of_features = X.shape[1]
    X_mat = X.values
    sequence_length = seq_len + 1
    datanew = []
    
    for index in range(len(X_mat) - sequence_length):
        datanew.append(X_mat[index: index + sequence_length])
    
    datanew = np.array(datanew)
    train_split = int(round(train_size * datanew.shape[0]))
    train_data = datanew[:train_split, :]
    
    X_train = train_data[:, :-1]
    y_train = train_data[:, -1][:,-1]
    
    X_test = datanew[train_split:, :-1] 
    y_test = datanew[train_split:, -1][:,-1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], amount_of_features))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], amount_of_features))  

    return X_train, y_train, X_test, y_test

window = 22
X['close'] = y
X_train, y_train, X_test, y_test = load_data(X, window)

model = Sequential()
model.add(LSTM(128, input_shape= (window, 6), return_sequences = True))
model.add(Dropout(0.2))

model.add(Conv1D(32, kernel_size=3, input_shape=(window, 6)))
model.add(Dropout(0.2))

model.add(LSTM(128, input_shape = (window, 6), return_sequences=False))
model.add(Dropout(0.2))

model.add(Dense(32))

model.add(Dense(16))

model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, batch_size=16, validation_split = 0.1, epochs = 32)

trainPredict = model.predict(X_train)
testPredict = model.predict(X_test)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([y_train])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([y_test])

r2_tr = 1 - np.sum((trainY[0] - trainPredict[:,0])**2) / np.sum((trainY[0] - np.mean(trainPredict[:,0]))**2)
print('r2 train: ',r2_tr)
r2_te = 1 - np.sum((testY[0] - testPredict[:,0])**2) / np.sum((testY[0] - np.mean(testPredict[:,0]))**2)
print('r2 test: ',r2_te)

plot_predicted = testPredict.copy()
plot_predicted = plot_predicted.reshape(487, 1)
plot_actual = testY.copy()
plot_actual = plot_actual.reshape(487, 1)

plot_predicted_train = trainPredict.copy()
plot_predicted_train = plot_predicted_train.reshape(1946, 1)
plot_actual_train = trainY.copy()
plot_actual_train = plot_actual_train.reshape(1946, 1)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

axes[0].plot(pd.DataFrame(plot_predicted_train), label='Train Predicted')
axes[0].plot(pd.DataFrame(plot_actual_train), label='Train Actual')
axes[0].legend(loc='best')

axes[1].plot(pd.DataFrame(plot_predicted), label='Test Predicted')
axes[1].plot(pd.DataFrame(plot_actual), label='Test Actual')
axes[1].legend(loc='best')

plt.show()