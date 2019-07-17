# Import modules
import numpy as np
import os
import pandas as pd
import sys
import tensorflow as tf
import time
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, Reshape
from tensorflow.keras.models import Model

# Remove hidden files from a list of files.
def removeHiddenFiles(list_files):
	counter = 0
	while True:
		if counter < len(list_files):
			if list_files[counter][0] == '.':
				list_files.pop(counter)
			elif len(list_files[counter]) < 7:
				list_files.pop(counter)
			elif list_files[counter][-1] != 'y':
				list_files.pop(counter)
			else:
				counter+= 1
		else:
			break

def Fc1Layer(flatX):
	X = Dense(5000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(flatX)
	X = Dense(26364, activation="linear")(X)
	return X

def Fc5Layer(flatX):
	X = Dense(4000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(flatX)
	X = Dense(3000, activation="tanh", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(2000, activation="tanh", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(3000, activation="tanh", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(4000, activation="tanh", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(26364, activation="linear")(X)
	return X

def ResNet1Layer(flatX):
	X = Dense(4000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(flatX)
	X = Dense(3000, activation="tanh", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(3000, activation="tanh", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(4000, activation="tanh", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(26364, activation="linear")(X)
	X = tf.keras.layers.add([flatX, X])
	return X

def ResNet3Layer(flatX):
	X = Dense(4000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(flatX)
	X = Dense(4000, activation="tanh", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(26364, activation="linear")(X)
	X = tf.keras.layers.add([flatX, X])

	X = Dense(3000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(3000, activation="tanh", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(26364, activation="linear")(X)
	X = tf.keras.layers.add([flatX, X])

	X = Dense(2000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(2000, activation="tanh", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(26364, activation="linear")(X)
	X = tf.keras.layers.add([flatX, X])

	return X

def DenseNet5Layer(flatX): #  MSE = 2.4782e-05 for 60 epochs 
	X = Dense(3000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(flatX)
	X = Dense(2000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(2000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(26364, activation="linear")(X)
	X = tf.keras.layers.add([flatX, X])

	X = Dense(2000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(1000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(1000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(26364, activation="linear")(X)
	X = tf.keras.layers.add([flatX, X])

	X = Dense(1000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(500, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(500, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(26364, activation="linear")(X)
	X = tf.keras.layers.add([flatX, X])

	X = Dense(2000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(1000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(1000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(26364, activation="linear")(X)
	X = tf.keras.layers.add([flatX, X])

	X = Dense(3000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(2000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(2000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(26364, activation="linear")(X)
	X = tf.keras.layers.add([flatX, X])

	return X

def ResNet5Layer(flatX): #  MSE = 2.4784e-05 for 60 epochs 
	X = Dense(3000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(flatX)
	X = Dense(2000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(2000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(26364, activation="linear")(X)
	X1 = tf.keras.layers.add([flatX, X])

	X = Dense(2000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X1)
	X = Dense(1000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(1000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(26364, activation="linear")(X)
	X2 = tf.keras.layers.add([X1, X])

	X = Dense(1000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X2)
	X = Dense(500, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(500, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(26364, activation="linear")(X)
	X3 = tf.keras.layers.add([X2, X])

	X = Dense(2000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X3)
	X = Dense(1000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(1000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(26364, activation="linear")(X)
	X4 = tf.keras.layers.add([X3, X])

	X = Dense(3000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X4)
	X = Dense(2000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(2000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(26364, activation="linear")(X)
	X5 = tf.keras.layers.add([X4, X])
	
	return X5

def DenseNet7Layer(flatX): #  MSE = 2.4788e-05 for 60 epochs  
	X = Dense(3000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(flatX)
	X = Dense(2000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(2000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(26364, activation="linear")(X)
	X = tf.keras.layers.add([flatX, X])

	X = Dense(2000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(1000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(1000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(26364, activation="linear")(X)
	X = tf.keras.layers.add([flatX, X])

	X = Dense(1000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(500, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(500, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(26364, activation="linear")(X)
	X = tf.keras.layers.add([flatX, X])

	X = Dense(800, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(400, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(400, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(26364, activation="linear")(X)
	X = tf.keras.layers.add([flatX, X])

	X = Dense(800, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(400, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(400, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(26364, activation="linear")(X)
	X = tf.keras.layers.add([flatX, X])

	X = Dense(2000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(1000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(1000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(26364, activation="linear")(X)
	X = tf.keras.layers.add([flatX, X])

	X = Dense(3000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(2000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(2000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(26364, activation="linear")(X)
	X = tf.keras.layers.add([flatX, X])

	return X

def ResNet7Layer(flatX): #  MSE = 2.4785e-05 for 60 epochs 
	X = Dense(3000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(flatX)
	X = Dense(2000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(2000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(26364, activation="linear")(X)
	X1 = tf.keras.layers.add([flatX, X])

	X = Dense(2000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X1)
	X = Dense(1000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(1000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(26364, activation="linear")(X)
	X2 = tf.keras.layers.add([X1, X])

	X = Dense(1000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X2)
	X = Dense(500, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(500, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(26364, activation="linear")(X)
	X3 = tf.keras.layers.add([X2, X])

	X = Dense(800, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X3)
	X = Dense(400, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(400, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(26364, activation="linear")(X)
	X4 = tf.keras.layers.add([X3, X])

	X = Dense(800, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X4)
	X = Dense(400, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(400, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(26364, activation="linear")(X)
	X5 = tf.keras.layers.add([X4, X])

	X = Dense(2000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X5)
	X = Dense(1000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(1000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(26364, activation="linear")(X)
	X6 = tf.keras.layers.add([X5, X])

	X = Dense(3000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X6)
	X = Dense(2000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(2000, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(l=lambd))(X)
	X = Dense(26364, activation="linear")(X)
	X7 = tf.keras.layers.add([X6, X])
	
	return X7

# Check for correct input
if len(sys.argv) < 3:
	print("Usage:")
	print("  python ./train.py [Training set directory] [Model File Name to Save] {Dev set directory}")
	sys.exit(0)


### Some hyperparameters are defined here
learningRate = 5e-6
lambd = 0.0000
numEpochs = 60
batchSize = 512
###

print("TensorFlow Version: " + tf.__version__)
print("Κeras Version: " + tf.keras.__version__)
# Save input parameters as variables
path = sys.argv[1]
modelPath = sys.argv[2]
if len(sys.argv)>=4:
	tPath = sys.argv[3]

# Do not show warnings
tf.logging.set_verbosity(tf.logging.ERROR)

### Start reading data files ###
tstart = time.time()

dataPath = path + '/data/'
labelPath = path + '/label/'

print('Directory Path: ' + path)
data_list = os.listdir(dataPath)
label_list = os.listdir(labelPath)

removeHiddenFiles(data_list)
removeHiddenFiles(label_list)

numFiles = len(data_list)
 
for i in range(numFiles):
	data_file= np.load(dataPath + data_list[0])
	label_file = np.load(labelPath + label_list[0])

label_file = np.reshape(label_file,(label_file.shape[0],label_file.shape[1]*label_file.shape[2]))


### If Dev directory was given, read test data files
test_data= None
test_labels = None
if len(sys.argv)==4:
	tDataPath = tPath + 'data/'
	tLabelPath = tPath + 'label/'

	tData_list = os.listdir(tDataPath)
	tLabel_list = os.listdir(tLabelPath)

	removeHiddenFiles(tData_list)
	removeHiddenFiles(tLabel_list)
	numTestFiles = len(tData_list)

	for i in range(numTestFiles):
		test_data = np.load(tDataPath + tData_list[0])
		test_labels = np.load(tLabelPath + tLabel_list[0])

	test_labels = np.reshape(test_labels,(test_labels.shape[0],test_labels.shape[1]*test_labels.shape[2]))

inputLayer = Input(shape=(8788,3))
fl1 = Flatten()(inputLayer)

#### REPLACE MODEL HERE ####
X = ResNet7Layer(fl1) # Hidden fc layer
############################

currModel = Model(inputs=inputLayer, outputs=X)
currModel.compile(optimizer = tf.train.AdamOptimizer(learning_rate = learningRate),
              loss = 'mse',
              metrics=['mse', 'mae'])
if len(sys.argv)==4:
	history = currModel.fit(data_file, label_file, epochs=numEpochs, batch_size=batchSize, validation_data=(test_data, test_labels))
else:
	history = currModel.fit(data_file, label_file, epochs=numEpochs, batch_size=batchSize)


hist_df = pd.DataFrame(history.history) 
hist_csv_file = 'Res7_' + 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)
modelPathCurr = "Res7_" + modelPath
print("Saving model as: " + modelPathCurr)
currModel.save(modelPathCurr)
print("Clearing current model for next model...")
tf.keras.backend.clear_session()
del currModel, history, hist_df

tend = time.time()
print('Elapsed Time: ' + str(tend - tstart))