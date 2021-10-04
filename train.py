import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.metrics import precision_recall_curve
import wandb
from wandb.keras import WandbCallback
from utils import DataGenerator
from models import build_CNN, build_LSTM

wandb.login()
print("tf version:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def main():
	dataset_path = 'data/npy_mfcc'
	train_click_dir  = os.path.join(dataset_path,'train','click')
	train_noClick_dir =  os.path.join(dataset_path,'train','noClick')
	test_click_dir   = os.path.join(dataset_path,'test','click')
	test_noClick_dir   = os.path.join(dataset_path,'test','noClick')

	train_click = sorted(os.listdir(train_click_dir))
	train_click = [os.path.join('train','click',ID) for ID in train_click]
	train_noClick = sorted(os.listdir(train_noClick_dir))
	train_noClick = [os.path.join('train','noClick',ID) for ID in train_noClick]
	test_click = sorted(os.listdir(test_click_dir))
	test_click = [os.path.join('test','click',ID) for ID in test_click]
	test_noClick = sorted(os.listdir(test_noClick_dir))
	test_noClick = [os.path.join('test','noClick',ID) for ID in test_noClick]

	partition = {'train': train_click[:12600] + train_noClick[:12600],
             	'validation': train_click[12600:] + train_noClick[12600:],
             	'test': test_click + test_noClick}
	labels = {}

	for i, ID in enumerate(partition['train']):
		labels[ID] = 0 if i < 12600 else 1 
	for i, ID in enumerate(partition['validation']):
		labels[ID] = 0 if i < 5400 else 1 
	for i, ID in enumerate(partition['test']):
		labels[ID] = 0 if i < 2000 else 1 

	ARCH = 'CNN'


	# Parameters

	# Start a new run, tracking hyperparameters in config
	wandb.init(project="LSTMvsCNN_clicks",
	           config={
	            "learning_rate": 0.0001,
	            "loss_function": "binary_crossentropy",
	            "batch_size": 32,
	            "epochs": 2,
	            "architecture": ARCH,
	            "dataset": dataset_path
	})
	config = wandb.config

	#############################################

	LSTM_params = {'dim': (501,),
	          'batch_size': config.batch_size,
	          'n_classes': 2,
	          'n_channels': 13,
	          'shuffle': True}
	LSTM_test_params = {'dim': (501,),
	          'batch_size': config.batch_size,
	          'n_classes': 2,
	          'n_channels': 13,
	          'shuffle': False}

	#############################################

	CNN_params = {'dim': (501,13),
	          'batch_size': config.batch_size,
	          'n_classes': 2,
	          'n_channels': 1,
	          'shuffle': True}
	CNN_test_params = {'dim': (501,13),
	          'batch_size': config.batch_size,
	          'n_classes': 2,
	          'n_channels': 1,
	          'shuffle': False}

	###############################################

	if ARCH == 'CNN':
	  	training_generator = DataGenerator(partition['train'], labels, ARCH='CNN',**CNN_params)
	  	validation_generator = DataGenerator(partition['validation'], labels, ARCH='CNN', **CNN_params)
	  	test_generator = DataGenerator(partition['test'], labels, ARCH='CNN', **CNN_test_params)
	  	# Design model
	  	model = build_CNN((501,13,1))
	elif ARCH == 'LSTM':
	  	training_generator = DataGenerator(partition['train'], labels, ARCH='LSTM', **LSTM_params)
	  	validation_generator = DataGenerator(partition['validation'], labels, ARCH='LSTM', **LSTM_params)
	  	test_generator = DataGenerator(partition['test'], labels, ARCH='LSTM', **LSTM_test_params)
	  	# Design model
	  	model = build_LSTM((501, 13))



	optimizers = tf.keras.optimizers.Adam(learning_rate=0.0001)
	model.compile(optimizer=optimizers,
	              loss='binary_crossentropy',
	              metrics=['accuracy',
	                       keras.metrics.Precision(),
	                       keras.metrics.Recall()])
	model.summary()

	# Train model on dataset
	# Note that colab first epoch on the kernal will be really slow. 
	# After the first epoch it should speed up because of cache
	history = model.fit(training_generator,
	                 	validation_data=validation_generator,
	                    epochs=config.epochs,
	                    use_multiprocessing=True,
	                    workers=6,
	                    callbacks=[WandbCallback()],
	                    verbose=1)	

if __name__ == "__main__":
	main()