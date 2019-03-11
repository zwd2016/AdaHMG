# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 20:36:34 2018

@author: WenDong Zheng
"""

from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D 
import matplotlib.pyplot as plt
from keras import optimizers

np.random.seed(10)
(x_Train,y_Train),(x_Test,y_Test) = mnist.load_data()
x_Train4D = x_Train.reshape(x_Train.shape[0],28,28,1).astype('float32')
x_Test4D = x_Test.reshape(x_Test.shape[0],28,28,1).astype('float32')

x_Train4D_normalize = x_Train4D / 255
x_Test4D_normalize = x_Test4D / 255
x_Test4D_normalize1 = x_Test4D / 255
x_Test4D_normalize2 = x_Test4D / 255

y_TrainOneHot = np_utils.to_categorical(y_Train)
y_TestOneHot = np_utils.to_categorical(y_Test)
y_TestOneHot1 = np_utils.to_categorical(y_Test)
y_TestOneHot2 = np_utils.to_categorical(y_Test)

#model-adam
model = Sequential()
model.add(Conv2D(filters=16,kernel_size=(5,5),padding='same',input_shape=(28,28,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=36,kernel_size=(5,5),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
train_history = model.fit(x=x_Train4D_normalize,y=y_TrainOneHot,validation_split=0.2,epochs=10,batch_size=300,verbose=2)

#model-adahmg
model1 = Sequential()
model1.add(Conv2D(filters=16,kernel_size=(5,5),padding='same',input_shape=(28,28,1),activation='relu'))
model1.add(MaxPooling2D(pool_size=(2,2)))
model1.add(Conv2D(filters=36,kernel_size=(5,5),padding='same',activation='relu'))
model1.add(MaxPooling2D(pool_size=(2,2)))
model1.add(Dropout(0.25))
model1.add(Flatten())
model1.add(Dense(128,activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(10,activation='softmax'))
print(model1.summary())
model1.compile(loss='categorical_crossentropy',optimizer='adahmg',metrics=['accuracy'])
train_history1 = model1.fit(x=x_Train4D_normalize,y=y_TrainOneHot,validation_split=0.2,epochs=10,batch_size=300,verbose=2)

#model-amsgrad
model2 = Sequential()
model2.add(Conv2D(filters=16,kernel_size=(5,5),padding='same',input_shape=(28,28,1),activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Conv2D(filters=36,kernel_size=(5,5),padding='same',activation='relu'))
model2.add(MaxPooling2D(pool_size=(2,2)))
model2.add(Dropout(0.25))
model2.add(Flatten())
model2.add(Dense(128,activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(10,activation='softmax'))
print(model2.summary())
adam = optimizers.Adam(amsgrad=True)
model2.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
train_history2 = model2.fit(x=x_Train4D_normalize,y=y_TrainOneHot,validation_split=0.2,epochs=10,batch_size=300,verbose=2)


plt.ylabel("Train loss value")  
plt.xlabel("The number of epochs")  
plt.title("Loss function-epoch curves")
plt.plot(train_history.history['loss'], label='train_Adam')
plt.plot(train_history2.history['loss'], label='train_Amsgrad')
plt.plot(train_history1.history['loss'], label='train_Adahmg')
plt.legend()
plt.show()


plt.ylabel("Validation Loss value")  
plt.xlabel("The number of epochs")  
plt.title("Loss function-epoch curves")
plt.plot(train_history.history['val_loss'], label='val_Adam')
plt.plot(train_history2.history['val_loss'], label='val_Amsgrad')
plt.plot(train_history1.history['val_loss'], label='val_Adahmg')
plt.legend()
plt.show()

# plot history train acc
plt.ylabel("Train Accuracy value")  
plt.xlabel("The number of epochs")  
plt.title("Accuracy-Epoch Curves")
plt.plot(train_history.history['acc'], label='train_Adam')
plt.plot(train_history2.history['acc'], label='train_Amsgrad')
plt.plot(train_history1.history['acc'], label='train_Adahmg')
plt.legend()
plt.show()

# plot history train acc
plt.ylabel("Validation Accuracy value")  
plt.xlabel("The number of epochs")  
plt.title("Accuracy-Epoch Curves")
plt.plot(train_history.history['val_acc'], label='val_Adam')
plt.plot(train_history2.history['val_acc'], label='val_Amsgrad')
plt.plot(train_history1.history['val_acc'], label='val_Adahmg')
plt.legend()
plt.show()

scores = model.evaluate(x_Test4D_normalize,y_TestOneHot)
print()
print('score_adam: %.6f' % scores[1])
print()

scores2 = model2.evaluate(x_Test4D_normalize2,y_TestOneHot2)
print()
print('score_amsgrad: %.6f' % scores2[1])

scores1 = model1.evaluate(x_Test4D_normalize1,y_TestOneHot1)
print()
print('score_adahmg: %.6f' % scores1[1])
