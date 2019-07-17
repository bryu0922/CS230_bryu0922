# Import modules
import numpy as np
import os
import sys
import tensorflow as tf
import time
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten
from tensorflow.keras.models import Model, load_model

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

# Check for correct input
if len(sys.argv) < 3:
	print("Usage:")
	print("  python ./test.py [Test set directory] [Model .h5 file path] ")
	sys.exit(0)

print("TensorFlow Version: " + tf.__version__)
print("Κeras Version: " + tf.keras.__version__)

# Save input parameters as variables
tPath = sys.argv[1]
modelPath = sys.argv[2]

# Load model
print("Loading model...")
currModel = load_model(modelPath)
print("Loading model complete!")
# Load test files
tDataPath = tPath + 'data/'
tLabelPath = tPath + 'label/'

# print('Test File Path: ' + tPath)
tData_list = os.listdir(tDataPath)
tLabel_list = os.listdir(tLabelPath)

removeHiddenFiles(tData_list)
removeHiddenFiles(tLabel_list)
numTestFiles = len(tData_list)

for i in range(numTestFiles):
	test_data = np.load(tDataPath + tData_list[0])
	test_lables = np.load(tLabelPath + tLabel_list[0])

test_lables = np.reshape(test_lables,(test_lables.shape[0],test_lables.shape[1]*test_lables.shape[2]))
currModel.compile(optimizer = tf.train.AdamOptimizer(learning_rate = 0.01),
              loss = 'mse',
              metrics=['accuracy', 'mse', 'mae'])
currModel.evaluate(test_data, test_lables)
tend = time.time()
