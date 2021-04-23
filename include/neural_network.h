#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>
#include <cmath>
#include <exception>
#include <memory>
#include <string>
#include <fstream>

#include "dataset.h"
#include "randomizer.h"

/* ----- Forward Declarations ----- */

class NeuralNetwork;
class Layer;

/* ----- Lossers ----- */

namespace lossers
{
	/*
	 * Identifies the method of the losser.
	 */
	enum Method
	{
		MSE
	};

	/*
	 * Losser defines a method for determining the loss (error) of a neural network w.r.t. a set
	 * of data points (outputs) and neural network predictions (results).
	 */
	class Losser
	{
	public:
		/*
		 * Determine the loss of the data points (outputs) versus the neural network predictions
		 * (results).
		 */
		virtual double loss(const Dataset& outputs, const Dataset& results) = 0;

		/*
		 * Derivative of the loss function.
		 */
		virtual Vector loss_derivative(const Vector& outputs, const Vector& results) = 0;

		/*
		 * Saves the losser to the specified file.
		 */
		void write_to_file(std::ofstream& outfile);

		// Identifies the losser method.
		const Method method;
	
	protected:
		/*
		 * Specifies derived losser method.
		 */
		Losser(Method m);
	};

	/*
	 * Creates a losser of the specified method.
	 */
	std::shared_ptr<Losser> make_losser(Method method);

	/*
	 * Creates a losser from the input file.
	 */
	std::shared_ptr<Losser> read_from_file(std::ifstream& infile);

	/*
	 * Losser that uses mean squared error.
	 */
	class MeanSquaredError : public Losser
	{
	public:
		MeanSquaredError();

		virtual double loss(const Dataset& outputs, const Dataset& results);
		virtual Vector loss_derivative(const Vector& outputs, const Vector& results);
	};
}

/* ----- Activators ------ */

namespace activators
{
	/*
	 * Identifies the method of the activator.
	 */
	enum Method
	{
		LINEAR,
		SIGMOID,
		RELU,
	};

	/*
	 * Activator defines a node activation function.
	 */
	class Activator
	{
	public:
		
		/*
		 * Activation function for a single node.
		 */
		virtual double activation(double x) = 0;

		/*
		 * Derivative of the activation function for a single node.
		 */
		virtual double activation_derivative(double x) = 0;

		/*
		 * Activation functions for a vector of nodes.
		 */
		Vector activation(Vector x);
		Vector activation_derivative(Vector x);

		/*
		 * Writes activator to the specified file.
		 */
		void write_to_file(std::ofstream& outfile);

		/*
		 * Method of the activator.
		 */
		const Method method;

	protected:
		/*
		 * Specifies derived activator method.
		 */
		Activator(Method m);
	};

	/*
	 * Constructs an activator of the specified method.
	 */
	std::unique_ptr<Activator> make_activator(Method method);

	/*
	 * Constructs an activator from the specified input file.
	 */
	std::unique_ptr<Activator> read_from_file(std::ifstream& infile);

	/*
	 * Linear activation method.
	 */
	class Linear : public Activator
	{
	public:
		Linear();
		virtual double activation(double x);
		virtual double activation_derivative(double x);
	};

	/*
	 * Sigmoid activation method.
	 */
	class Sigmoid : public Activator
	{
	public:
		Sigmoid();
		virtual double activation(double x);
		virtual double activation_derivative(double x);
	};

	/*
	 * Rectified Linear Unit method.
	 */
	class ReLU : public Activator
	{
	public:
		ReLU();
		virtual double activation(double x);
		virtual double activation_derivative(double x);
	};
}

/* ----- Optimizers ----- */

namespace optimizers
{
	/*
	 * Specifies optimizer method.
	 */
	enum Method
	{
		SGD,
		ADAM,
	};

	/*
	 * Optimizer defines a method for optimizing a neural network's loss function.
	 */
	class Optimizer
	{
	public:
		/*
		 * Perform a single optimization of a neural network's weights and thresholds
		 * for the specified dataset.
		 */
		virtual void optimize(NeuralNetwork& network, const Dataset& dataset) = 0;

		/*
		 * Return a list of this optimizer's parameters.
		 */
		virtual std::vector<double> get_parameters() const = 0;

		/*
		 * Print the optimizer.
		 */
		void print() const;

		/*
		 * Write the optimizer to the specified file.
		 */
		virtual void write_to_file(std::ofstream& outfile);
	
		/*
		 * Specifies the optimizer's method.
		 */
		const Method method;

	protected:
		/*
		 * Construct an optimizer for a network of the specified layout and method.
		 */
		Optimizer(const std::vector<size_t>& layout, Method m);

		/*
		 * Adds d_weights and d_thresholds to the respective network weights and thresholds.
		 */
		void apply_changes(NeuralNetwork& network);

		/*
		 * Calculates the gradient of the thresholds and weights via back-propagation.
		 */
		void calculate_gradient(NeuralNetwork& network, const Dataset& dataset);

		/*
		 * Changes to weights and thresholds the optimizer determines.
		 */
		std::vector<Vector> d_thresholds;
		std::vector<Matrix> d_weights;

		/*
		 * Gradient of the weights and thresholds.
		 */
		std::vector<Vector> thresholds_g;
		std::vector<Matrix> weights_g;
	};
	
	/*
	 * Construct an optimizer from the specified method, layout, and parameters.
	 */
	std::unique_ptr<Optimizer> make_optimizer(Method method, std::vector<size_t> layout, std::vector<double> parameters);

	/*
	 * Construct an optimizer with the given layout and from the specified input file.
	 */
	std::unique_ptr<Optimizer> read_from_file(std::ifstream& infile, std::vector<size_t> layout);

	/*
	 * Performs optimization via stochastic gradient descent.
	 * Parameters:
	 * 	alpha - momentum factor
	 * 	eta - learning rate
	 */
	class StochasticGradientDescent : public Optimizer
	{
	public:
		StochasticGradientDescent(const std::vector<size_t>& layout, const std::vector<double>& parameters);

		virtual void optimize(NeuralNetwork& network, const Dataset& dataset);
		virtual std::vector<double> get_parameters() const;
	
	private:
		double alpha;	// Momentum
		double eta;	// Learning rate

		/*
		 * Applies momentum to the previous weight & threshold changes.
		 */
		void apply_momentum();
	};

	/*
	 * Adaptive Momentum Estimation (Adam)
	 * Parameters:
	 * 	alpha - learning rate
	 * 	beta1 - first moment forgetful factor
	 * 	beta2 - second moment forgetful factor
	 */
	class Adam : public Optimizer
	{
	public:
		Adam(const std::vector<size_t>& layout, const std::vector<double>& parameters);

		virtual void optimize(NeuralNetwork& network, const Dataset& dataset);
		virtual std::vector<double> get_parameters() const;

	private:
		double alpha;		// Learning rate
		double beta_1, beta_2;	// Forgetful factors
		double t;		// Timestep

		/*
		 * Calculate moments for
		 * 	g - gradient
		 * 	m - first moment
		 * 	v - second moment
		 */
		double moment_calc(double g, double& m, double& v);

		/*
		 * First moment running averages
		 */
		std::vector<Matrix> weights_m;
		std::vector<Vector> thresholds_m;

		/*
		 * Second moment running averages
		 */
		std::vector<Matrix> weights_v;
		std::vector<Vector> thresholds_v;
	};
};

/* ----- Layer ----- */

/*
 * Layer defines the operation of a single layer of a neural network.
 */
class Layer
{
public:
	/*
	 * Construct a layer with:
	 * 	input_size - number of input nodes
	 * 	output_size - number of output nodes
	 * 	a - activator
	 */
	Layer(size_t input_size, size_t output_size, std::unique_ptr<activators::Activator> a);

	/*
	 * Copy layer
	 * Requires explicit implementation to "copy" activators.
	 * 
	 * (Destructor not specified since all memory safely allocated).
	 */
	Layer(const Layer& layer);
	Layer& operator=(const Layer& layer);

	/*
	 * Set layer's nodes to the input vector.
	 */
	Layer& operator=(const Vector& v);

	/*
	 * Computes the outputs of the layer from the nodes values.
	 */
	Vector compute() const;

	/*
	 * Print the vector.
	 */
	void print() const;

	/*
	 * Randomize the weights via the specified randomizer.
	 */
	void randomize_weights(Randomizer& r);

	/*
	 * Writes the layer to the specified output file.
	 */
	void write_to_file(std::ofstream& outfile);

	/*
	 * Read and construct a layer from the specified input file.
	 */
	static Layer read_from_file(std::ifstream& infile);

	/*
	 * Layer data
	 */
	Vector nodes;
	Vector thresholds;
	Matrix weights;
	std::unique_ptr<activators::Activator> activator;

	/*
	 * Linear combination of weights * nodes + thresholds
	 */
	Vector h() const;
};

/* ----- Neural Network ----- */

/*
 * A single feed-forward neural network.
 */
class NeuralNetwork
{
public:
	/*
	 * Build a neural network with the parameters:
	 * 	layout - node layout
	 * 	activations - activation method of each layer
	 * 	losser - losser method
	 * 	op - optimizer method
	 */
	NeuralNetwork(std::vector<size_t> layout, std::vector<activators::Method> activations, lossers::Method l, std::unique_ptr<optimizers::Optimizer> op);

	/*
	 * Copy a neural network from an existing one.
	 */
	NeuralNetwork(const NeuralNetwork& network);
	NeuralNetwork& operator=(const NeuralNetwork& network);

	/*
	 * Read a neural network from the specified file.
	 */
	NeuralNetwork(const std::string& filename);

	/*
	 * Forward propagate the inputs through the neural network.
	 */
	Vector compute(const Vector& inputs);

	/*
	 * Perform a single training iteration from the given dataset.
	 */
	void back_propagate(const Dataset& dataset);

	/*
	 * Print the neural network.
	 */
	void print() const;

	/*
	 * Randomize weights of the neural network.
	 */
	void randomize_weights();

	/*
	 * Calculate the loss of the neural network.
	 */
	double loss(const Dataset& outputs, const Dataset& results);

	/*
	 * Returns a schematic of the layout of the neural network.
	 */
	std::vector<size_t> get_layout() const;

	/*
	 * Write the neural network to the file.
	 */
	void write_to_file(const std::string& filename);

	/*
	 * Read the neural network from a file.
	 */
	void read_from_file(const std::string& filename);

	/*
	 * Neural network parameters.
	 */
	std::vector<Layer> layers;
	std::shared_ptr<lossers::Losser> losser;
	std::unique_ptr<optimizers::Optimizer> optimizer;
};




#endif
