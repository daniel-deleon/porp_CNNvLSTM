import tensorflow as tf
import tensorflow.keras as keras

def build_CNN(input_shape):
  
	# create model
  	model = keras.Sequential()

  	# 1st conv layer
  	model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
  	model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same'))
  	model.add(keras.layers.BatchNormalization())

  	# 2nd conv layer
  	model.add(keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape))
  	model.add(keras.layers.MaxPool2D((3,3), strides=(2,2), padding='same'))
  	model.add(keras.layers.BatchNormalization())

  	# 3rd conv layer
  	model.add(keras.layers.Conv2D(32, (2,2), activation='relu', input_shape=input_shape))
  	model.add(keras.layers.MaxPool2D((2,2), strides=(2,2), padding='same'))
  	model.add(keras.layers.BatchNormalization())

  	# flatten the output and feed it into dense layer
  	model.add(keras.layers.Flatten())
  	model.add(keras.layers.Dense(64, activation='relu'))
  	model.add(keras.layers.Dropout(0.3))

  	# output layer
  	model.add(keras.layers.Dense(1,activation='sigmoid'))

  	return model

def build_LSTM(input_shape):
  	# create model
  	model = keras.Sequential()

  	# 2 LSTM layers
  	model.add(keras.layers.LSTM(64, input_shape=input_shape, return_sequences=True))
  	model.add(keras.layers.LSTM(64))

  	#dense layer
  	model.add(keras.layers.Dense(64, activation='relu'))
  	model.add(keras.layers.Dropout(0.3))

  	# output layer
  	model.add(keras.layers.Dense(1,activation='sigmoid'))

  	return model