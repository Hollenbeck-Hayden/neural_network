# A Toy Neural-Network Program


This program was a way to gain some experience with neural networks (which I haven't used since
high school). My master's thesis involves fitting functions to data via feed-forward nerual networks,
so I replicated some of the basic techniques here.


## Dependencies


The program is built against C++17. There are no other dependencies.


## Build


Starting at the parent directory, create a build fold and enter it

```bash
mkdir build && cd build
```

Run cmake on the parent directory, then make the program

```bash
cmake ..
make
```

Run the program.

```bash
./neural_network
```

Results are saved to the `data` directory.


## Design


The core structure of the program is the `NeuralNetwork` class, which is built from modular components.
This allows the user to bind different loss methods, activation functions, and optimization methods
at runtime. Developers may also extend the base classes and provide extra functionality with minimal overhead.
The neural network itself is sectioned into fully connected layers, where each layer feeds directly into
the next. All nodes in a layer share the same activation function (can be modified, but is an uncommon
use case). k-fold cross validation is also implemented in the main file.

The network was optimized to avoid unnecessary copies and allocations where possible. This was done
by using in-place modification algorithms and move semantics. As practise, I used the standard algorithms
library (and related techniques) perhaps too much, but it was quite fun to write!


## Implemented Components

### Loss Methods
	* Mean Squared Error (MSE)

### Activation Functions
	* Linear (LINEAR)
	* Sigmoidal (SIGMOID)
	* Rectified Linear Unit (RELU)

### Optimization Methods
	* Stochastic Gradient Descent (SGD)
	* Adaptive Moment Estimation (ADAM)


## Examples

Tests of the implementation against a linear and quadratic function are included in the `examples`
folder. There's also a python plotting script which can recreate various plots for different outputs.
Because of the smaller amount of data and only one data point per abscissa, the network will tend
to overfit these cases (but at least you know it acts like a neural network). 


Linear Function
	- Linear function with error offset at each point.
	- Neural network with mean squared error loss, ADAM optimization
	- Layers: 1-5-5-1, Sigmoid-Sigmoid-Linear
	- 4-fold cross validation

Quadratic Function
	- Quadratic function with error offset at each point.
	- Neural network with mean squared error loss, ADAM optimization
	- Layers: 1-5-5-1, Sigmoid-Sigmoid-Linear
	- 5-fold cross validation
