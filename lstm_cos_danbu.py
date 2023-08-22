from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import matplotlib.pyplot as plt

dataset = np.cos(np.arange(1000)*(20*np.pi/1000))

plt.plot(dataset)
plt.show()

##以下的create_dataset函数可以重复调用
def create_dataset(dataset, lookback=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-lookback):
        dataX.append(dataset[i:(i+lookback)])
        dataY.append(dataset[i+lookback])
    return np.array(dataX), np.array(dataY)

lookback = 3 #用lookback预测1天

train_size = int(len(dataset)*0.7)
test_size = len(dataset) - train_size
train, test = dataset[:train_size], dataset[train_size:]

trainX, trainY = create_dataset(train, lookback)
testX, testY = create_dataset(test, lookback)

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

print('trainX.shape = ', trainX.shape)
print('testX.shape = ', testX.shape)
print('trainY.shape = ', trainY.shape)
print('testY.shape = ', testY.shape)

model = Sequential()
model.add(LSTM(32, input_shape=(lookback, 1)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(trainX, trainY, batch_size=32, epochs=10)

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredictPlot = np.zeros(shape=(len(dataset), 1))
trainPredictPlot[:] = np.nan #nan表示缺失值NAN
trainPredictPlot[lookback:len(trainPredict)+lookback, :] = trainPredict

testPredictPlot = np.zeros(shape=(len(dataset), 1))
testPredictPlot[:] = np.nan
testPredictPlot[len(trainPredict)+lookback:len(dataset)-lookback, :] = testPredict

plt.plot(dataset, label='origin')
plt.plot(trainPredictPlot, label='trainPredict')
plt.plot(testPredictPlot, label='testPredict')
plt.legend(loc='upper right')
plt.show()

