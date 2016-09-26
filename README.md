# NucFrames

Utilities to help with analysing multiple single cell structures.
Precomputes distances / depths for each .nuc file. 
Helper functions for dealing with offset starts etc.

A NucFrame object represents the structure for a single cell Hi-C experiment. It
can be created from a .nuc file with the from_nuc method. This will create an
hdf5 file with various cached results (for instance, depth). Alternatively, this
file can be directly loaded with NucFrame.\_\_init\_\_.

A NucFrames object loads multiple NucFrame objects and ensures they have
consistent start/end basepairs. It is created by passing in a list of NucFrame
file locations, typically done with glob.glob("/path/to/files/*.hdf5").

## Installation

Requires python3. Using Anaconda to create a fresh environment, and then manage
these requirements, is suggested. All packages can be installed with pip or
conda.

Run ```python setup.py install``` to install.

### Python Packages

* numpy
* scipy
* numba (>= 0.29.0, installable via ```conda install -c https://conda.anaconda.org/numba numba```)
* h5py
* tqdm
* networkx
* pandas
* hypothesis

## Running

Example analysis scripts using this library can be found [here](https://github.com/latkins/SingleCellStructureAnalysis).
