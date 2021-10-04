import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
class DataGenerator(tf.keras.utils.Sequence):
  	'Generates data for Keras'
  	def __init__(self, list_IDs, labels, batch_size=32, dim=(501,13),
  	            n_channels=1, n_classes=2, shuffle=True, ARCH='CNN'):
  	    'Initialization'
  	    self.dim = dim
  	    self.batch_size = batch_size
  	    self.labels = labels
  	    self.list_IDs = list_IDs
  	    self.n_channels = n_channels
  	    self.n_classes = n_classes
  	    self.shuffle = shuffle
  	    self.arch = ARCH
  	    self.on_epoch_end()

  	def on_epoch_end(self):
  	  	'Updates indexes after each epoch'
  	  	self.indexes = np.arange(len(self.list_IDs))
  	  	if self.shuffle == True:
  	  	  np.random.shuffle(self.indexes)

  	def __data_generation(self, list_IDs_temp):
  	  	'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
  	  	# Initialization
  	  	X = np.empty((self.batch_size, *self.dim, self.n_channels))
  	  	y = np.empty((self.batch_size), dtype=int)

  	  	# Generate data
  	  	for i, ID in enumerate(list_IDs_temp):
  	  	    # Store sample
  	  	    if self.arch == 'LSTM':
  	  	      X[i,] = np.load(os.path.join('data/npy_mfcc', ID))
  	  	    elif self.arch == 'CNN':
  	  	      X[i,] = np.expand_dims(np.load(os.path.join('data/npy_mfcc', ID)), axis=2)
  	  	    # Store class
  	  	    y[i] = self.labels[ID]

  	  	#return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
  	  	return X, y
  
  	def __len__(self):
  	  	'Denotes the number of batches per epoch'
  	  	return int(np.floor(len(self.list_IDs) / self.batch_size))


  	def __getitem__(self, index):
  	  	'Generate one batch of data'
  	  	# Generate indexes of the batch
  	  	indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
	

  	  	# Find list of IDs
  	  	list_IDs_temp = [self.list_IDs[k] for k in indexes]

  	  	# Generate data
  	  	X, y = self.__data_generation(list_IDs_temp)
  	  	return X, y


def plot_and_save_history(history, save_dir):
  
  	np.save(os.path.join(save_dir, 'metrics', 'history.npy'), history.history)

  	#fig, axs = plt.subplots(2)
  	plt.figure(figsize=(15,10))
  	fig1 = plt.subplot(2,1,1)
  	# create accuracy subplot
  	plt.plot(history.history['accuracy'],label='train accuracy')
  	plt.plot(history.history['val_accuracy'],label='val accuracy')
  	plt.grid()
  	fig1.set_ylabel('Accuracy')
  	fig1.legend(loc='lower right')
  	fig1.set_title('Accuracy')
  	fig1.set_ylim(top=1)

  	fig2 = plt.subplot(2,1,2)
  	# create error subplot
  	plt.plot(history.history['loss'],label='train loss')
  	plt.plot(history.history['val_loss'],label='val loss')
  	plt.grid()
  	fig2.set_ylabel('Loss')
  	fig2.set_xlabel('Epoch')
  	fig2.legend(loc='upper right')
  	fig2.set_title('Loss')
  	fig2.set_ylim(bottom=0)

  	plt.savefig(os.path.join(save_dir, 'loss_acc.png'))
  	plt.show()

def plot_and_save_PR_curve(model, test_generator, save_dir):
  	predictions = model.predict(test_generator)
  	test_labels = np.concatenate((np.zeros(2000), np.ones(2000)))
  	precision, recall, thresholds = precision_recall_curve(test_labels, predictions)

  	np.save(os.path.join(save_dir, 'metrics', 'PR.npy'), {'precision': precision,
  	                                           'recall': recall})

  	plt.figure(figsize=(15,10))
  	plt.plot(recall,precision)
  	plt.title('PR Curve')
  	plt.ylabel('Precision')
  	plt.xlabel('Recall')
  	plt.grid()
  	plt.savefig(os.path.join(save_dir, 'PR.png'))
  	plt.show()