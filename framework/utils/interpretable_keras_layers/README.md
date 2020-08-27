Explicitly Argumentative and Relational Keras Layers
==========

## Installation

* requires python 3.7+
* requires tensorflow 2.2.0

## Run Demo
* cd to the repo
* run `python3 test.py` for an example on MNIST dataset

## Usage in your script
* add the following line at the beginning of your script `import ExplicitlyArgumentativeLayer, ExplicitlyRelationalLayer, OR,NOR,AND,NAND,XOR,XNOR` and then call `ExplicitlyArgumentativeLayer(object_pairs=16, edge_size_per_object_pair=4, operators_set=[OR,NOR,AND,NAND,XOR,XNOR], argument_links=8)` or `ExplicitlyRelationalLayer(object_pairs=16, edge_size_per_object_pair=4, operators_set=[OR,NOR,AND,NAND,XOR,XNOR])` inside a keras model 
* play with the parameters of the two layers, thus changing the performance of the model

**Tested on macOS Mojave and Catalina**

## Contact

To report issues, use GitHub Issues. 
For other queries, contact Francesco Sovrano: 
* <francesco.sovrano2@unibo.it>
* <cesco.sovrano@gmail.com>
