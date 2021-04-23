#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include "neural_network.h"

/* ----- Dataset Parameters ----- */

static const double X_LOW = -5;
static const double X_HIGH = 5;
static const double DATA_STEP = 0.2;
static const double PRED_STEP = 0.1;

/* ----- Data Utility Functions ----- */

/*
 * Target function for the neural network to match
 */
Vector target_function(const Vector& x)
{
	Vector v(1);
	v[0] = -3.0 + x[0];
	return v;
}

/*
 * Generates data points with random error at each point.
 */
Dataset generate_data()
{
	Randomizer rand(-1, 1);
	Dataset data = generate_range(X_LOW, X_HIGH, DATA_STEP);
	operations::modify(data.begin(), data.end(),
		[r = &rand] (Point<Vector>& p) -> void {
			p.y = target_function(p.x);
			p.y[0] += r->next();
		});

	return data;
}

/*
 * Generates results of the neural network for a given input set.
 */
Dataset generate_results(NeuralNetwork& nn, const Dataset& inputs)
{
	Dataset data(inputs.size());
	operations::modify(data.begin(), data.end(), inputs.begin(),
		[n = &nn] (Point<Vector>& p, const Point<Vector>& q) -> void {
			p.x = q.x;
			p.y = n->compute(p.x);
		});

	return data;
}

/*
 * Generates predictions interpolated by the neural network.
 */
Dataset generate_predictions()
{
	return generate_range(X_LOW, X_HIGH, PRED_STEP);
}

/*
 * Write the dataset and interpolated function to a file.
 */
void write_to_file(const std::string& filename, const Dataset& data, const Dataset& predictions)
{
	std::ofstream out(filename);

	out << data.size() << std::endl;
	for (const auto& p : data)
		out << p.x[0] << ", " << p.y[0] << std::endl;
	
	out << predictions.size() << std::endl;
	for (const auto& p : predictions)
		out << p.x[0] << ", " << p.y[0] << std::endl;

	out.close();
}

/*
 * Writes validation data to a file specified by i.
 */
void write_xvalidation_to_file(size_t i, const std::vector<double>& training_error, const std::vector<double>& validation_error)
{
	std::stringstream sstream;
	sstream << "../data/xvalid_" << i << ".data";
	std::ofstream outfile(sstream.str());
	outfile << training_error.size() << std::endl;
	for (int i = 0; i < training_error.size(); i++)
		outfile << training_error[i] << ", " << validation_error[i] << std::endl;
	outfile.close();
}

/* ----- Training Methods ----- */

/*
 * Trains a network for the specified number of epochs.
 */
void train_network_epoch(NeuralNetwork& network, const Dataset& data, size_t n_epochs)
{
	for (size_t t = 0; t < n_epochs; t++)
		network.back_propagate(data);
}

/*
 * Performs k-fold cross-validation training against the ith fold.
 */
void cross_validation(NeuralNetwork& network, const Dataset& data, size_t k, size_t i)
{
	std::cout << "Training " << i << "th neural network on " << k << "-fold dataset" << std::endl;
	size_t subset_size = data.size() / k;

	const size_t MAX_EPOCHS = 200; // Number of epochs to run
	const size_t EPOCH_SIZE = 500; // Number of iterations to run in an epoch (reduces number of validation checks)

	// Get training & validation sets
	const Dataset training   = data.get_training_set  (i, subset_size);
	const Dataset validation = data.get_validation_set(i, subset_size);

	// Store errors for visualization
	std::vector<double>   training_error(MAX_EPOCHS);
	std::vector<double> validation_error(MAX_EPOCHS);

	// Network that has best validation error
	NeuralNetwork best_network = (const NeuralNetwork&) network;
	double best_error = best_network.loss(validation, generate_results(network, validation));

	// Iterate over all epochs
	for (size_t epoch = 0; epoch < MAX_EPOCHS; epoch++)
	{
		// Train the network
		train_network_epoch(network, training, EPOCH_SIZE);

		// Record errors
		  training_error[epoch] = network.loss(  training, generate_results(network,   training));
		validation_error[epoch] = network.loss(validation, generate_results(network, validation));

		// Check if this is the best network
		if (validation_error[epoch] < best_error)
		{
			best_network = (const NeuralNetwork&) network;
			best_error = validation_error[epoch];
		}
	}

	// Set network as the best one
	network = std::move(best_network);

	// Save to file
	write_xvalidation_to_file(i, training_error, validation_error);
}

/* ----- Main ----- */

int main(void)
{
	/* ----- Set network parameters ----- */
	size_t layer_size = 5;
	const std::vector<size_t> layout{1, layer_size, layer_size, 1};
	const std::vector<activators::Method> activations{activators::Method::SIGMOID, activators::SIGMOID, activators::Method::LINEAR};
	const optimizers::Method optimizer_method = optimizers::Method::ADAM;
	const std::vector<double> optimizer_parameters{0.0001, 0.9, 0.999};
	
	/* ----- Build / Load Dataset ----- */
	const Dataset data = generate_data();
	//const Dataset data = Dataset::read_from_file("../data/test.data");

	/* ----- Cross Validation ----- */
	const size_t k = 4;			// k-fold cross validation
	std::vector<NeuralNetwork> networks;

	// Randomize data order to get distributed folds
	const Dataset rand_ordered_data = randomize_dataset_order(data);
	rand_ordered_data.write_to_file("../data/test_rand.data");

	for (size_t i = 0; i < k; i++)
	{
		// Build a network for ith fold
		std::unique_ptr<optimizers::Optimizer> optimizer = optimizers::make_optimizer(optimizer_method, layout, optimizer_parameters);
		NeuralNetwork network(layout, activations, lossers::Method::MSE, std::move(optimizer));
		network.randomize_weights();

		// Perform cross validation
		cross_validation(network, rand_ordered_data, k, i);

		// Save network
		networks.push_back(std::move(network));
	}

	/* ----- Output results of the neural network ----- */

	Dataset avg = generate_predictions();
	operations::modify(avg.begin(), avg.end(),
		[n = layout.back()] (Point<Vector>& p) -> void {
			p.y = Vector(n);
		});

	for (size_t i = 0; i < k; i++)
	{
		std::stringstream sstream;
		sstream << "../data/results_" << i << ".data";
		const Dataset preds = generate_results(networks[i], generate_predictions());
		write_to_file(sstream.str(), data, preds);

		operations::modify(avg.begin(), avg.end(), preds.begin(), [n = (double) k] (Point<Vector>& a, const auto& b) -> void { a.y += b.y / k; });
	}

	write_to_file("../data/results_average.data", data, avg);

	return 0;
}
