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
	v[0] = -3.0 + x[0] * x[0];
	return v;
}

/*
 * Generates data points with random error at each point.
 */
Dataset generate_data()
{
	Dataset data;
	data.x = generate_range(X_LOW, X_HIGH, DATA_STEP);
	data.y = std::vector<Vector>(data.x.size());

	Randomizer rand(-1, 1);

	for (size_t i = 0; i < data.x.size(); i++)
	{
		data.y[i] = target_function(data.x[i]);
		data.y[i][0] += rand.next();
	}

	return data;
}

/*
 * Generates results of the neural network for a given input set.
 */
Dataset generate_results(NeuralNetwork& nn, const std::vector<Vector>& inputs)
{
	Dataset data;
	data.x = inputs;
	data.y = std::vector<Vector>(data.x.size());

	for (size_t i = 0; i < data.x.size(); i++)
		data.y[i] = nn.compute(data.x[i]);
	
	return data;
}

/*
 * Generates predictions interpolated by the neural network.
 */
Dataset generate_predictions(NeuralNetwork& nn)
{
	return generate_results(nn, generate_range(X_LOW, X_HIGH, PRED_STEP));
}

/*
 * Write the dataset and interpolated function to a file.
 */
void write_to_file(const std::string& filename, const Dataset& data, const Dataset& predictions)
{
	std::ofstream out(filename);

	out << data.x.size() << std::endl;
	for (int i = 0; i < data.x.size(); i++)
		out << data.x[i][0] << ", " << data.y[i][0] << std::endl;
	
	out << predictions.x.size() << std::endl;
	for (int i = 0; i < predictions.x.size(); i++)
		out << predictions.x[i][0] << ", " << predictions.y[i][0] << std::endl;

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

	const size_t MAX_EPOCHS = 100;
	const size_t EPOCH_SIZE = 20;

	const Dataset training   = data.get_training_set  (i, subset_size);
	const Dataset validation = data.get_validation_set(i, subset_size);

	std::vector<double>   training_error(MAX_EPOCHS);
	std::vector<double> validation_error(MAX_EPOCHS);

	std::vector<NeuralNetwork> networks;
	networks.reserve(MAX_EPOCHS);

	for (size_t epoch = 0; epoch < MAX_EPOCHS; epoch++)
	{
		train_network_epoch(network, training, EPOCH_SIZE);
		  training_error[epoch] = network.loss(  training.y, generate_results(network,   training.x).y);
		validation_error[epoch] = network.loss(validation.y, generate_results(network, validation.x).y);
		networks.push_back((const NeuralNetwork&) network);
	}

	size_t mindex = 0;
	for (size_t i = 0; i < validation_error.size(); i++)
		if (validation_error[i] < validation_error[mindex])
			mindex = i;
	
	network = std::move(networks[mindex]);

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
	const size_t k = 5;			// k-fold cross validation
	std::vector<NeuralNetwork> networks;

	const Dataset rand_ordered_data = randomize_dataset_order(data);
	rand_ordered_data.write_to_file("../data/test_rand.data");

	for (size_t i = 0; i < k; i++)
	{
		std::unique_ptr<optimizers::Optimizer> optimizer = optimizers::make_optimizer(optimizer_method, layout, optimizer_parameters);
		NeuralNetwork network(layout, activations, lossers::Method::MSE, std::move(optimizer));
		network.randomize_weights();

		cross_validation(network, rand_ordered_data, k, i);

		networks.push_back(std::move(network));
	}

	/* ----- Output results of the neural network ----- */

	Dataset dummy_preds = generate_predictions(networks.front());
	Dataset avg(dummy_preds.size());
	for (size_t i = 0; i < avg.size(); i++)
	{
		avg.x[i] = dummy_preds.x[i];
		avg.y[i] = Vector(dummy_preds.y[i]);
		avg.y[i].clear();
	}

	for (size_t i = 0; i < k; i++)
	{
		std::stringstream sstream;
		sstream << "../data/results_" << i << ".data";
		const Dataset preds = generate_predictions(networks[i]);
		write_to_file(sstream.str(), data, preds);

		for (size_t j = 0; j < avg.size(); j++)
		{
			avg.y[j] += preds.y[j] / ((double) k);
		}
	}

	write_to_file("../data/results_average.data", data, avg);

	return 0;
}
