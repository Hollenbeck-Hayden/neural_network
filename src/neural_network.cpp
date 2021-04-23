#include "neural_network.h"

namespace lossers
{
	/* ----- Losser ----- */

	Losser::Losser(Method m)
		: method(m)
	{}

	void Losser::write_to_file(std::ofstream& outfile)
	{
		outfile << (int) method << std::endl;
	}

	/* ----- MeanSquaredError ----- */

	MeanSquaredError::MeanSquaredError()
		: Losser(Method::MSE)
	{}

	double MeanSquaredError::loss(const std::vector<Vector>& outputs, const std::vector<Vector>& results)
	{
		double total = 0;
		for (size_t i = 0; i < outputs.size(); i++)
			for (size_t j = 0; j < outputs[i].length(); j++)
				total += (outputs[i][j] - results[i][j]) * (outputs[i][j] - results[i][j]);
		return 0.5 * total;
	}

	Vector MeanSquaredError::loss_derivative(const Vector& outputs, const Vector& results)
	{
		return results - outputs;
	}

	/* ----- Losser Creation ----- */

	std::shared_ptr<Losser> make_losser(Method method)
	{
		switch (method)
		{
			case MSE:
			default:
				return std::make_shared<MeanSquaredError>();
		};
	}

	std::shared_ptr<Losser> read_from_file(std::ifstream& infile)
	{
		int method;
		infile >> method;
		return make_losser((Method) method);
	}

}

namespace activators
{
	/* ----- Activator ----- */

	Activator::Activator(Method m)
		: method(m)
	{}

	Vector Activator::activation(Vector x)
	{
		for (size_t i = 0; i < x.length(); i++)
			x[i] = this->activation(x[i]);
		return x;
	}

	Vector Activator::activation_derivative(Vector x)
	{
		for (size_t i = 0; i < x.length(); i++)
			x[i] = this->activation_derivative(x[i]);
		return x;
	}

	void Activator::write_to_file(std::ofstream& outfile)
	{
		outfile << (int) method << std::endl;
	}

	/* ----- Linear ----- */

	Linear::Linear()
		: Activator(Method::LINEAR)
	{}

	double Linear::activation(double x)
	{
		return x;
	}

	double Linear::activation_derivative(double x)
	{
		return 1;
	}

	/* ----- Sigmoid ----- */

	Sigmoid::Sigmoid()
		: Activator(Method::SIGMOID)
	{}

	double Sigmoid::activation(double x)
	{
		return 1.0 / (1.0 + exp(-x));
	}

	double Sigmoid::activation_derivative(double x)
	{
		double a = exp(-x);
		return a / ((1.0 + a) * (1.0 + a));
	}

	/* ----- ReLU ----- */

	ReLU::ReLU()
		: Activator(Method::RELU)
	{}

	double ReLU::activation(double x)
	{
		return std::max(0.0, x);
	}

	double ReLU::activation_derivative(double x)
	{
		return (x < 0) ? 0 : 1;
	}

	/* ----- Activator Creation ----- */
	std::unique_ptr<Activator> make_activator(Method method)
	{
		switch (method)
		{
			case Method::LINEAR:
				return std::make_unique<Linear>();

			case Method::SIGMOID:
				return std::make_unique<Sigmoid>();

			case Method::RELU:
				return std::make_unique<ReLU>();

			default:
				return std::make_unique<Linear>();
		}
	}

	std::unique_ptr<Activator> read_from_file(std::ifstream& infile)
	{
		int method;
		infile >> method;
		return make_activator((Method) method);
	}
}

namespace optimizers
{
	/* ----- Optimizer ----- */

	Optimizer::Optimizer(const std::vector<size_t>& layout, Method m)
		: method(m)
	{
		d_weights.reserve(layout.size()-1);
		weights_g.reserve(d_weights.size());

		d_thresholds.reserve(layout.size()-1);
		thresholds_g.reserve(d_thresholds.size());

		for (size_t i = 0; i < layout.size()-1; i++)
		{
			d_weights.push_back(Matrix(layout[i+1], layout[i]));
			d_thresholds.push_back(Vector(layout[i+1]));

			weights_g.push_back(Matrix(d_weights[i].num_rows(), d_weights[i].num_cols()));
			thresholds_g.push_back(Vector(d_thresholds[i].length()));
		}
	}

	void Optimizer::print() const
	{
		for (size_t l = 0; l < d_weights.size(); l++)
		{
			std::cout << "layer " << l << std::endl;
			std::cout << "d_weights:\n";
			print_matrix(d_weights[l]);
			std::cout << "d_thresh:\n";
			print_vec(d_thresholds[l]);
			std::cout << "\n";
		}
	}

	void Optimizer::write_to_file(std::ofstream& outfile)
	{
		const std::vector<double> parameters = this->get_parameters();
		outfile << (int) method << std::endl;
		outfile << parameters.size() << std::endl;
		for (size_t i = 0; i < parameters.size(); i++)
			outfile << parameters[i] << " ";
		outfile << std::endl;
	}

	void Optimizer::apply_changes(NeuralNetwork& network)
	{
		for (size_t l = 0; l < network.layers.size(); l++)
		{
			network.layers[l].thresholds += d_thresholds[l];
			network.layers[l].weights += d_weights[l];
		}
	}

	void Optimizer::calculate_gradient(NeuralNetwork& network, const Dataset& dataset)
	{
		// Clear previous gradient entries
		for (size_t l = 0; l < weights_g.size(); l++)
		{
			weights_g[l].clear();
			thresholds_g[l].clear();
		}

		// Calculate gradient over sum of data points
		for (size_t a = 0; a < dataset.size(); a++)
		{
			Vector results = network.compute(dataset.x[a]);
			Vector Delta = network.losser->loss_derivative(dataset.y[a], results);
			
			for (int index = network.layers.size()-1; index >= 0; index--)
			{
				Delta *= network.layers[index].activator->activation_derivative(network.layers[index].h());

				for (size_t i = 0; i < weights_g[index].num_rows(); i++)
					for (size_t j = 0; j < weights_g[index].num_cols(); j++)
						weights_g[index](i,j) += Delta[i] * network.layers[index].nodes[j];

				thresholds_g[index] += Delta;

				Delta = Delta * network.layers[index].weights;
			}
		}
	}

	/* ----- StochasticGradientDescent ----- */

	StochasticGradientDescent::StochasticGradientDescent(const std::vector<size_t>& layout, const std::vector<double>& parameters)
		: Optimizer(layout, Method::SGD)
	{
		eta = parameters[0];
		alpha = parameters[1];
	}

	void StochasticGradientDescent::optimize(NeuralNetwork& network, const Dataset& dataset)
	{
		apply_momentum();

		calculate_gradient(network, dataset);

		for (size_t l = 0; l < weights_g.size(); l++)
			d_weights[l] -= eta * weights_g[l];

		for (size_t l = 0; l < thresholds_g.size(); l++)
			d_thresholds[l] -= eta * thresholds_g[l];

		apply_changes(network);
	}

	void StochasticGradientDescent::apply_momentum()
	{
		for (size_t i = 0; i < d_weights.size(); i++)
		{
			d_weights[i] *= alpha;
			d_thresholds[i] *= alpha;
		}
	}

	std::vector<double> StochasticGradientDescent::get_parameters() const
	{
		return std::vector<double>{eta, alpha};
	}

	/* ----- Adam ----- */

	Adam::Adam(const std::vector<size_t>& layout, const std::vector<double>& parameters)
		: Optimizer(layout, Method::ADAM)
	{
		alpha = parameters[0];
		beta_1 = parameters[1];
		beta_2 = parameters[2];

		weights_m.reserve(d_weights.size());
		weights_v.reserve(d_weights.size());
		for (size_t i = 0; i < d_weights.size(); i++)
		{
			weights_m.push_back(Matrix(d_weights[i].num_rows(), d_weights[i].num_cols()));
			weights_v.push_back(Matrix(d_weights[i].num_rows(), d_weights[i].num_cols()));
		}

		thresholds_m.reserve(d_thresholds.size());
		thresholds_v.reserve(d_thresholds.size());
		for (size_t i = 0; i < d_thresholds.size(); i++)
		{
			thresholds_m.push_back(Vector(d_thresholds[i].length()));
			thresholds_v.push_back(Vector(d_thresholds[i].length()));
		}
	}

	void Adam::optimize(NeuralNetwork& network, const Dataset& dataset)
	{
		clear_mv_vectors();

		for (int t = 1; t < 60; t++)
		{
			calculate_gradient(network, dataset);

			for (size_t l = 0; l < d_weights.size(); l++)
				for (size_t i = 0; i < d_weights[l].num_rows(); i++)
					for (size_t j = 0; j < d_weights[l].num_cols(); j++)
						d_weights[l](i,j) = moment_calc(weights_g[l](i,j), weights_m[l](i,j), weights_v[l](i,j), (double) t);
			
			for (size_t l = 0; l < d_thresholds.size(); l++)
				for (size_t i = 0; i < d_thresholds[l].length(); i++)
					d_thresholds[l][i] = moment_calc(thresholds_g[l][i], thresholds_m[l][i], thresholds_v[l][i], (double) t);

			apply_changes(network);
		}
	}

	void Adam::clear_mv_vectors()
	{
		for (size_t i = 0; i < d_weights.size(); i++)
		{
			weights_m[i].clear();
			weights_v[i].clear();
		}
		
		for (size_t i = 0; i < d_thresholds.size(); i++)
		{
			thresholds_m[i].clear();
			thresholds_v[i].clear();
		}
	}


	double Adam::moment_calc(double g, double& m, double& v, double t)
	{
		const double epsilon = 1e-6;

		m = beta_1 * m + (1.0 - beta_1) * g;
		v = beta_2 * v + (1.0 - beta_2) * g * g;

		double mhat = m / (1.0 - pow(beta_1, t));
		double vhat = v / (1.0 - pow(beta_2, t));

		return -alpha * mhat / (sqrt(vhat) + epsilon);
	}

	std::vector<double> Adam::get_parameters() const
	{
		return std::vector<double>{alpha, beta_1, beta_2};
	}


	/* ----- Optimizer Creation ----- */

	std::unique_ptr<Optimizer> make_optimizer(Method method, std::vector<size_t> layout, std::vector<double> parameters)
	{
		switch (method)
		{
			case Method::ADAM:
				return std::make_unique<Adam>(layout, parameters);

			case Method::SGD:
			default:
				return std::make_unique<StochasticGradientDescent>(layout, parameters);
		}
	}

	
	std::unique_ptr<Optimizer> read_from_file(std::ifstream& infile, std::vector<size_t> layout)
	{
		int method;
		infile >> method;
		int num_params;
		infile >> num_params;

		std::vector<double> params(num_params);
		for (int i = 0; i < num_params; i++)
			infile >> params[i];

		return make_optimizer((Method) method, layout, params);
	}
}

/* ----- Layer ----- */

Layer::Layer(size_t input_size, size_t output_size, std::unique_ptr<activators::Activator> a)
	: nodes(input_size), weights(output_size, input_size), thresholds(output_size),
	  activator(std::move(a))
{}

Layer::Layer(const Layer& layer)
	: nodes(layer.nodes), weights(layer.weights), thresholds(layer.thresholds), activator(activators::make_activator(layer.activator->method))
{}

Layer& Layer::operator=(const Layer& layer)
{
	nodes = layer.nodes;
	weights = layer.weights;
	thresholds = layer.thresholds;
	activator = activators::make_activator(layer.activator->method);
	return *this;
}

Vector Layer::compute()
{
	return activator->activation(h());
}

Vector Layer::h() const
{
	return weights * nodes + thresholds;
}

void Layer::print() const
{
	std::cout << "Weights: " << std::endl;
	print_matrix(weights);

	std::cout << std::endl << "Thresholds: " << std::endl;
	print_vec(thresholds);

	std::cout << std::endl;
}

void Layer::randomize_weights(Randomizer& r)
{
	for (size_t i = 0; i < weights.num_rows(); i++)
		for (size_t j = 0; j < weights.num_cols(); j++)
			weights(i,j) = r.next();
}

void Layer::write_to_file(std::ofstream& outfile)
{
	activator->write_to_file(outfile);

	outfile << weights.num_rows() << " " << weights.num_cols() << std::endl;
	for (size_t i = 0; i < weights.num_rows(); i++)
	{
		for (size_t j = 0; j < weights.num_cols(); j++)
			outfile << weights(i,j) << " ";

		outfile << std::endl;
	}

	for (size_t i = 0; i < thresholds.length(); i++)
		outfile << thresholds[i] << " ";
	outfile << std::endl << std::endl;
}

Layer Layer::read_from_file(std::ifstream& infile)
{
	auto activator = activators::read_from_file(infile);
	size_t input_size, output_size;
	infile >> output_size >> input_size;

	Layer layer(input_size, output_size, std::move(activator));

	for (size_t i = 0; i < layer.weights.num_rows(); i++)
		for (size_t j = 0; j < layer.weights.num_cols(); j++)
			infile >> layer.weights(i,j);

	for (size_t i = 0; i < layer.thresholds.length(); i++)
		infile >> layer.thresholds[i];

	return layer;
}

/* ----- Neural Network ----- */

NeuralNetwork::NeuralNetwork(const std::string& filename)
{
	read_from_file(filename);
}

NeuralNetwork::NeuralNetwork(std::vector<size_t> layout, std::vector<activators::Method> activations, lossers::Method l, std::unique_ptr<optimizers::Optimizer> op)
	: losser(lossers::make_losser(l)), optimizer(std::move(op))
{
	if (activations.size() != layout.size()-1)
	{
		std::cout << "ERROR INCORRECT NN LAYOUT" << std::endl;
		throw std::exception();
	}

	layers.reserve(layout.size()-1);
	for (size_t i = 0; i < layout.size()-1; i++)
		layers.push_back(Layer(layout[i], layout[i+1], activators::make_activator(activations[i])));
}

Vector NeuralNetwork::compute(const Vector& inputs)
{
	layers.front().nodes = inputs;

	for (size_t i = 1; i < layers.size(); i++) {
		layers[i].nodes = layers[i-1].compute();
	}

	return layers.back().compute();
}

void NeuralNetwork::back_propagate(const Dataset& dataset)
{
	optimizer->optimize(*this, dataset);
}

void NeuralNetwork::print() const
{
	for (size_t i = 0; i < layers.size(); i++)
	{
		std::cout << "Layer " << i << std::endl;
		layers[i].print();
	}
}

void NeuralNetwork::randomize_weights()
{
	Randomizer r(0.001, 0.01);
	for (size_t i = 0; i < layers.size(); i++)
		layers[i].randomize_weights(r);
}

double NeuralNetwork::loss(const std::vector<Vector>& outputs, const std::vector<Vector>& results)
{
	return losser->loss(outputs, results);
}

void NeuralNetwork::write_to_file(const std::string& filename)
{
	std::ofstream outfile(filename);

	outfile << layers.size() << std::endl;
	for (size_t i = 0; i < layers.size(); i++)
	{
		layers[i].write_to_file(outfile);
	}

	losser->write_to_file(outfile);
	optimizer->write_to_file(outfile);

	outfile.close();
}

void NeuralNetwork::read_from_file(const std::string& filename)
{
	std::ifstream infile(filename);

	int layer_count;
	infile >> layer_count;

	layers = std::vector<Layer>();
	layers.reserve(layer_count);

	for (size_t i = 0; i < layer_count; i++)
	{
		layers.push_back(Layer::read_from_file(infile));
	}

	std::vector<size_t> layout(layer_count+1);
	for (size_t i = 0; i < layers.size(); i++)
		layout[i] = layers[i].nodes.length();

	layout.back() = layers.back().weights.num_rows();

	losser = lossers::read_from_file(infile);
	optimizer = optimizers::read_from_file(infile, layout);

	infile.close();
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& network)
	: layers(network.layers)
{
	optimizer = optimizers::make_optimizer(network.optimizer->method, network.get_layout(), network.optimizer->get_parameters());
	losser = lossers::make_losser(network.losser->method);
}

NeuralNetwork& NeuralNetwork::operator=(const NeuralNetwork& network)
{
	layers = network.layers;
	optimizer = optimizers::make_optimizer(network.optimizer->method, network.get_layout(), network.optimizer->get_parameters());
	losser = lossers::make_losser(network.losser->method);
	return *this;
}

std::vector<size_t> NeuralNetwork::get_layout() const
{
	std::vector<size_t> layout(layers.size()+1);
	for (size_t i = 0; i < layers.size(); i++)
		layout[i] = layers[i].nodes.length();
	layout.back() = layers.back().weights.num_rows();
	return layout;
}

