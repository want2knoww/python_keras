from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.datasets import imdb
from keras.utils import pad_sequences
from keras.layers import LSTM,Bidirectional,Dropout,Conv1D,MaxPooling1D

max_features = 20000
maxlen = 100
embedding_size = 128

kernel_size = 5
filters = 64
pool_size = 4

lstm_outpuy_size = 70

batch_size = 30
epochs = 2


print('loading data……')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(x_train.shape)
print(len(x_train), 'train_sequences')
print(len(x_test), 'test_sequences')
print('pad sequence (samples x time)')
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model…')
model = Sequential()
model.add(Embedding(max_features,embedding_size,input_length=maxlen)) #这里不用管输入序列长度
model.add(Dropout(0.25))
model.add(Conv1D(filters,kernel_size,padding='valid',activation='relu',strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_outpuy_size))
model.add(Dense(1,activation='sigmoid'))

print(model.summary())

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print('Train…')
model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,validation_data=(x_test,y_test))

score,acc = model.evaluate(x_test, y_test, batch_size=batch_size)

print('Test score:',score)
print('Test accuracy:',acc)