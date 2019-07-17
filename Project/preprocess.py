# Import modules
import sys
import subprocess
import time
import os
import math
import numpy as np
import random
'''
preprocess.py
'''
# Remove hidden files from a list of files.
def removeHiddenFiles(list_files):
	counter = 0
	while True:
		if counter < len(list_files):
			if list_files[counter][0] == '.':
				list_files.pop(counter)
			elif len(list_files[counter]) < 7:
				list_files.pop(counter)
			else:
				counter+= 1
		else:
			break


# Check for correct input
if len(sys.argv) < 3:
	print("Usage:")
	print("  python ./preprocess.py [Data directory] [Number of particles]")
	sys.exit(0)

# Save input parameters as variables
path = sys.argv[1]
number_particles = int(sys.argv[2])
print('Data files directory: ' + path)
print
# Obtain a list of all files and remove hidden files already in the data directory
list_files = os.listdir(path)
removeHiddenFiles(list_files)

# Create data and label directories, if they are not already there
dataPath = path + '/data/'
labelPath = path + '/label/'
try:
    os.mkdir(dataPath)
except OSError:
    pass
try:
    os.mkdir(labelPath)
except OSError:
    pass

# Create empty NumPy array to populate data
dataArray = np.zeros((len(list_files), number_particles,3))
labelArray = np.zeros((len(list_files), number_particles,3))

# Start time and start processing
tstart = time.time()
print("Number of files to process: " + str(len(list_files)))
for i in range(0, len(list_files)):
	file_to_read = list_files[i]

	# Report progress
	if (100*i % round(len(list_files)) == 0):
		progress = (i*1.0)/len(list_files)*100
		print('Current progress: ' + str(round(progress)) + '%')

	#Open file data file
	data_read = open(path + file_to_read, 'r')

	# Read simulation parameters
	line = data_read.readline()
	line = data_read.readline()
	current_timestep = int(line)
	current_timestep /= 100000

	line = data_read.readline()
	line = data_read.readline()
	num_particles = int(line)

	line = data_read.readline()
	line = data_read.readline()
	words = line.split()
	box_size = float(words[1])

	# Use NumPy genfromtxt to read data and label. 
	xParticles = np.genfromtxt(path + file_to_read, skip_header=9, max_rows = num_particles, usecols = (0,2,3,4),)
	yParticles = np.genfromtxt(path + file_to_read, skip_header=18 + num_particles, max_rows = num_particles, usecols = (0,2,3,4),)


	# Sort by particle ID so particles IDs can be omitted.
	xParticles = xParticles[xParticles[:,0].argsort()]
	yParticles = yParticles[yParticles[:,0].argsort()]

	# Normalize (x,y,z) positions with respect to box size, and shift.
	xParticles -= 0.5*box_size
	xParticles /= box_size
	yParticles -= 0.5*box_size
	yParticles /= box_size	

	# Remove particle ID
	dataArray[i,:,:] = xParticles[:,1:]
	labelArray[i,:,:] = yParticles[:,1:]

	data_read.close()

#Save Data and Label Files
np.save(dataPath + 'DataFile', dataArray)
np.save(labelPath + 'LabelFile', labelArray)
tend = time.time()
#Report time
print('Elapsed Time: ' + str(tend - tstart))