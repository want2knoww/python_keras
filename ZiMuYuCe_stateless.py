import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.utils import np_utils

alphabet = 'abcdefghijklmnopqrstuvwxyz'

#建立字母和数字之间的对应关系，其中c代表字母，i代表数字
char_to_int = dict((c,i) for i,c in enumerate(alphabet))  #enmuerate函数生成字符串与整数的对应关系
int_to_char = dict((i,c) for i,c in enumerate(alphabet))

#生成数据集
seq_length = 1  #seg_length就是look_back
dataX = []
dataY = []
for i in range(0,len(alphabet)-seq_length,1):
    seq_in = alphabet[i:i+seq_length]
    seq_out = alphabet[i+seq_length]
    dataX.append([char_to_int[char] for char in seq_in])
    dataY.append(char_to_int[seq_out])
    print(seq_in, '->', seq_out)

X = np.reshape(dataX, (len(dataX), seq_length, 1))
#归一化
X = X/ float(len(alphabet))

#独特码编码
Y = np_utils.to_categorical(dataY)

model = Sequential()
model.add(LSTM(32,input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(Y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X, Y, epochs=300, batch_size=1, verbose=1)

scores = model.evaluate(X, Y,verbose=0)
print('model accuracy: %.2f%%' % (scores[1]*100))

for pattern in dataX:
    x = np.reshape(pattern, (1, len(pattern), 1))
    x = x/float(len(alphabet))
    prediction = model.predict(x, verbose=0)  #输出包括26个值，每个代表出现该字母的概率
    index = np.argmax(prediction) #np.argmax确定最大值所在的索引
    result = int_to_char[index]
    seq_in = [int_to_char[value] for value in pattern]
    print(seq_in, '->', result)

