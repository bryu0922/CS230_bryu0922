# Future Prediction in Brownian Dynamics Simulations Using Deep Neural Networks
CS230 Spring 2019 Final Project
Last Updated 06/09/2019

Author: Brian K. Ryu

Note: Only one data file is currently uploaded, inside ./One/

## Requirements
This code was developed using default settings for the Amazon AWS Deep Learning AMI (Ubuntu) Version 22.0 - ami-01a4e5be5f289dd12

This includes MXNet-1.4, TensorFlow-1.13, PyTorch-1.0, Keras-2.2, Chainer-5.3, Caffe/2-0.8, Theano-1.0 & CNTK-2.6, configured with NVIDIA CUDA, cuDNN, NCCL, Intel MKL-DNN, Docker & NVIDIA-Docker. Upon instantiating an instance, the TensorFlow environment must be activated via
```
source activate tensorflow_p36
```
## Task
Given an [LAMMPS][LAMMPS] simulation snapshot (.lammpstrj) at a timestep, predict a future snapshot 10<sup>5</sup> timesteps later.

Detailed description can be viewed at the Appendix


## Description of Data Set
[LAMMPS][LAMMPS] (Large-scale Atomic/Molecular Massively Parallel Simulator) is an open-source molecular dynamics software that widely used for scientific research. LAMMPS "trajectory files" (.lammpstrj), which are used as inputs and outputs for the neural networks of this project, are simulation "snapshots" containing information to a current simulation system. These trajectory files typically contain the following information:
* Number of atoms/particles in simulation box/system
* Dimensions of simulation box
* Current timestep of simulation
* Unique ID and particle type (representing different molecules like O<sub>2</sub> or H<sub>2</sub>O, typically in integer numbers) of each atom/particle in simulation system
* (x,y,z) coordinates of each atom/particle in simulation system
* Linear and angular velocities of each atom/particle
* Charge, dipole moment, and etc. information pertaining to each particle

For this project, I will use minimal trajectory files only containing particle IDs, types, and positions (x,y,z). The first several lines look like the following:

```
ITEM: TIMESTEP
19400000
ITEM: NUMBER OF ATOMS
8788
ITEM: BOX BOUNDS pp pp pp
0.0000000000000000e+00 3.0024785703928551e+01
0.0000000000000000e+00 3.0024785703928551e+01
0.0000000000000000e+00 3.0024785703928551e+01
ITEM: ATOMS id type xu yu zu 
2578 2 15.7291 30.3692 0.621521 
8139 4 16.7065 0.135016 30.5353 
2122 2 18.7036 0.103409 0.411795 
720 5 25.6909 0.0309724 0.404852 
1346 1 26.5239 30.6178 0.492683 
...
...
```
Each .lammpstrj file contains two snapshots in a single file that are used as inputs and labels, and are 200000 timesteps separated apart.

The number of particles can vary significantly depending on the goal of simulation and 1 million particles is commonly used for massively parallelized simulations. Here, I choose a constant (8788) number of particles to limit unnecessary computational work in developing and testing various architectures.

The data set contains 14,400 .lammpstrj snapshots, generated from 200 independent simulations at 72 different time points (200x72=14,400). The dataset is divded into training, dev, and test sets as:
* Train: 12,000
* Dev: 1,200
* Test: 1,200

## Preprocessing
The preprocessing code is run via
```
python ./preprocess.py [Data directory] [Number of particles]
```
For the current data set, 8788 should be used in place of [Number of particles].

The preprocessing code reads the input and output from each data file and consolidates all data files into a DataFile.npy and LabelFile.npy files. For larger datasets other methods of data storages are preferred. 

Note that since the particle positions are sorted by particle ID, the particle ID is redundant information and is not saved. For the simulations used to generate the training, dev, and test sets the particle types (1-5) are all identical, so particle types were omitted as well.

## Training
The training script is run via
```
python ./train.py [Training set directory] [Model File Name to Save] {Dev set directory}
```
The training script loads the data, trains the model, and saves the trained model as an hdf5 format to the file name specified. The directory of the training data set (identical as [Data directory] from preprocessing), and the model file name (e.g. myModel.h5) must be specified.

Optionally adding a Dev set directory allows computation of cross-validation error that is computed with the dev set that was not used to train the model.

As of now, the current model are:
* FC1: One fully connected hidden layer with 5,000 units.
* FC5: Five fully connected hidden layers with 4,000, 3,000, 2,000, 3,000 and 4,000 units.
* ResNet1: A network with one residual block, containing four fully connected layers with 4,000, 3,000, 3,000, and 4,000 units in each layer.
* ResNet3: A network with three residual blocks, containing four fully connected layers.
* ResNet5: A network with five residual blocks, containing four fully connected layers.
* ResNet7: A network with seven residual blocks, containing four fully connected layers.

## Testing
The testing script is run via
```
python ./test.py [Test set directory] [Model .h5 file path]
```
The usage is identical to the train.py in which the file directory for the test set data and the model file name (e.g. myModel.h5) must be specified.

[LAMMPS]: https://lammps.sandia.gov/