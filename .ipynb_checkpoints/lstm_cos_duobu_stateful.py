from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import matplotlib.pyplot as plt

dataset = np.cos(np.arange(1000)*(20*np.pi/1000))

plt.plot(dataset)
plt.show()

def to_supervised(train, n_input, n_out=3):
    data = train
    x, y = list(), list()
    in_start = 0

    for _ in range(len(data)):
        in_end = in_start +n_input
        out_end = in_end + n_out
        if out_end < len(data):
            x_input = data[in_start:in_end]
            x_input = x_input.reshape((len(x_input), 1))
            x.append((x_input))
            y.append(data[in_end:out_end])
        in_start += 1
    return np.asarray(x), np.asarray(y)


train_size = int(len(dataset)*0.7)
test_size = len(dataset) - train_size
train, test = dataset[:train_size], dataset[train_size:]

trainX, trainY = to_supervised(train, n_input=7)
testX, testY = to_supervised(test, n_input=7)

verbose, epochs, batch_size = 2, 10, 1
n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainY.shape[1]

print('trainX.shape = ', trainX.shape)
print('testX.shape = ', testX.shape)
print('trainY.shape = ', trainY.shape)
print('testY.shape = ', testY.shape)

model = Sequential()
model.add(LSTM(32, input_shape=(n_timesteps, n_features), stateful=True, batch_input_shape=(batch_size, 7, 1)))  #batch_input_size=(batch_size, n_timesteps, n_features)
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs))

model.compile(loss='mse', optimizer='adam')
model.fit(trainX, trainY, batch_size=batch_size, epochs=epochs, verbose=verbose, shuffle=False)

n_input = 7

input_x = testX[-1]
print(input_x)
print(input_x.shape)

input_x = input_x.reshape((1, len(input_x), 1))   #需与lstm层的训练输入批次形状完全相同
print(input_x)
print(input_x.shape)


yhat = model.predict(input_x, verbose=2)
print(yhat)

