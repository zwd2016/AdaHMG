# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 15:43:41 2018

@author: WenDong Zheng
"""

from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import metrics
from keras import regularizers
from keras import optimizers
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
# load dataset
dataset = read_csv('pollution_pm2.5.csv', header=0, index_col=0)
values = dataset.values
# integer encode direction
encoder = LabelEncoder()
values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
# frame as supervised learning
reframed = series_to_supervised(scaled, 1, 1)
# drop columns we don't want to predict
reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
print(reframed.head())
# split into train and test sets
values = reframed.values
n_train_hours = 548 * 24#365*24*2=2years,548*24=1.5years
train = values[:n_train_hours, :]
test = values[n_train_hours:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
test_X1, test_y1 = test[:, :-1], test[:, -1]
test_X2, test_y2 = test[:, :-1], test[:, -1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
test_X1 = test_X1.reshape((test_X1.shape[0], 1, test_X1.shape[1]))
test_X2 = test_X2.reshape((test_X2.shape[0], 1, test_X2.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network optimizer=adam
model = Sequential()
model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam',metrics=['mae'])
# fit network
history = model.fit(train_X, train_y, epochs=20, batch_size=120, validation_data=(test_X, test_y), verbose=2, shuffle=False)

# design network optimizer=adahmg
model1 = Sequential()
model1.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2])))
model1.add(Dense(1))
model1.compile(loss='mae', optimizer='adahmg',metrics=['mae'])
# fit network
history1 = model1.fit(train_X, train_y, epochs=20, batch_size=120, validation_data=(test_X1, test_y1), verbose=2, shuffle=False)

# design network optimizer=amsgrad
model2 = Sequential()
model2.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2])))
model2.add(Dense(1))
adam = optimizers.Adam(amsgrad=True)
model2.compile(loss='mae', optimizer='adam',metrics=['mae'])
# fit network
history2 = model2.fit(train_X, train_y, epochs=20, batch_size=120, validation_data=(test_X2, test_y2), verbose=2, shuffle=False)

# plot history train-loss
pyplot.ylabel("Train loss value")  
pyplot.xlabel("The number of epochs")  
pyplot.title("Loss function-epoch curves")
pyplot.plot(history.history['loss'], label='train_adam')
pyplot.plot(history2.history['loss'], label='train_amsgrad')
pyplot.plot(history1.history['loss'], label='train_adahmg')
pyplot.legend()
pyplot.show()

# plot history val-loss
pyplot.ylabel("Validation Loss value")  
pyplot.xlabel("The number of epochs")  
pyplot.title("Loss function-epoch curves")
pyplot.plot(history.history['val_loss'], label='val_adam')
pyplot.plot(history2.history['val_loss'], label='val_amsgrad')
pyplot.plot(history1.history['val_loss'], label='val_adahmg')
pyplot.legend()
pyplot.show()

# make a prediction adam
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# make a prediction adahmg
yhat1 = model1.predict(test_X1)
test_X1 = test_X1.reshape((test_X1.shape[0], test_X1.shape[2]))

# make a prediction amsgrad
yhat2 = model2.predict(test_X2)
test_X2 = test_X2.reshape((test_X2.shape[0], test_X2.shape[2]))

# invert scaling for forecast adahmg
inv_yhat1 = concatenate((yhat1, test_X1[:, 1:]), axis=1)
inv_yhat1 = scaler.inverse_transform(inv_yhat1)
inv_yhat1 = inv_yhat1[:,0]

# invert scaling for forecast amsgrad
inv_yhat2 = concatenate((yhat2, test_X2[:, 1:]), axis=1)
inv_yhat2 = scaler.inverse_transform(inv_yhat2)
inv_yhat2 = inv_yhat2[:,0]

# invert scaling for forecast adam
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]

# invert scaling for actual adam
inv_y = scaler.inverse_transform(test_X)
inv_y = inv_y[:,0]

# invert scaling for actual adahmg
inv_y1 = scaler.inverse_transform(test_X1)
inv_y1 = inv_y1[:,0]

# invert scaling for actual amsgrad
inv_y2 = scaler.inverse_transform(test_X2)
inv_y2 = inv_y2[:,0]

# calculate RMSE adam
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Adam Test RMSE: %.3f' % rmse)

# calculate RMSE amsgrad
rmse2 = sqrt(mean_squared_error(inv_y2, inv_yhat2))
print('Amsgrad Test RMSE: %.3f' % rmse2)

# calculate RMSE AdaHMG
rmse1 = sqrt(mean_squared_error(inv_y1, inv_yhat1))
print('AdaHMG Test RMSE: %.3f' % rmse1)

pyplot.title('PM 2.5(the next 24 hours)')
pyplot.xlabel('Time range(h)')
pyplot.ylabel(' PM2.5 range')
pyplot.plot(inv_y[:24],label='true_adam')
pyplot.plot(inv_yhat[:24],'r--',label='predictions_adam')
pyplot.plot(inv_yhat2[:24],'p--',label='predictions_amsgrad')
pyplot.plot(inv_yhat1[:24],'g--',label='predictions_adahmg')
pyplot.legend()
pyplot.show()