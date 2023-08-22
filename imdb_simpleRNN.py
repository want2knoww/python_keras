
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.datasets import imdb
from keras.utils import pad_sequences
from keras.layers import SimpleRNN

max_features = 20000
maxlen = 80
batch_size = 64

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
model.add(Embedding(max_features,128)) #这里不用管输入序列长度
model.add(SimpleRNN(128,dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(1,activation='sigmoid'))
print(model.summary())

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

print('Train…')
model.fit(x_train,y_train,batch_size=batch_size,epochs=15,validation_data=(x_test,y_test))

score,acc = model.evaluate(x_test, y_test, batch_size=batch_size)

print('Test score:',score)
print('Test accuracy:',acc)