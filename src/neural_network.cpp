#include "neural_network.h"
#include "operations.h"

#include <algorithm>
#include <numeric>

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

	double MeanSquaredError::loss(const Dataset& outputs, const Dataset& results)
	{
		return 0.5 * std::transform_reduce(outputs.begin(), outputs.end(), results.begin(), 0.0,
				std::plus<double>(),
				[] (const auto& a, const auto& b) -> double {
					const Vector c = a.y - b.y;
					return dot(c,c);
				});
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
		// Initialize corresponding difference and gradient vectors for each layer
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
		operations::modify(network.layers.begin(), network.layers.end(), d_thresholds.begin(), [] (Layer& layer, const Vector& threshold) -> void { layer.thresholds += threshold; });
		operations::modify(network.layers.begin(), network.layers.end(),    d_weights.begin(), [] (Layer& layer, const Matrix&    weight) -> void { layer.weights    +=    weight; });
	}

	void Optimizer::calculate_gradient(NeuralNetwork& network, const Dataset& dataset)
	{
		// Clear previous gradient entries
		std::for_each(   weights_g.begin(),    weights_g.end(), [] (Matrix& m) -> void { m.clear(); });
		std::for_each(thresholds_g.begin(), thresholds_g.end(), [] (Vector& v) -> void { v.clear(); });

		// Calculate gradient over sum of data points
		for (const auto& data_point : dataset)
		{
			Vector results = network.compute(data_point.x);
			Vector Delta = network.losser->loss_derivative(data_point.y, results);
			
			// Back-propagate through the layers
			auto layer	 = network.layers.rbegin();
			auto weight_g	 = weights_g.rbegin();
			auto threshold_g = thresholds_g.rbegin();

			while(layer != network.layers.rend())
			{
				// Multiply Delta components by activation derivatives at each node
				Delta *= layer->activator->activation_derivative(layer->h());

				// Add tensor product of Delta and layer's nodes to the gradient
				for (size_t i = 0; i < weight_g->num_rows(); i++)
					for (size_t j = 0; j < weight_g->num_cols(); j++)
						(*weight_g)(i,j) += Delta[i] * layer->nodes[j];

				// Add Delta to threshold gradient
				(*threshold_g) += Delta;

				// Prep delta for next layer
				// May change Delta's size depending on weight dimensions
				Delta = Delta * layer->weights;

				// Increment iterators
				layer++;
				weight_g++;
				threshold_g++;
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

		operations::modify(   d_weights.begin(),    d_weights.end(),    weights_g.begin(), [eta = this->eta] (Matrix& a, const Matrix& b) -> void { a -= eta * b; });
		operations::modify(d_thresholds.begin(), d_thresholds.end(), thresholds_g.begin(), [eta = this->eta] (Vector& a, const Vector& b) -> void { a -= eta * b; });

		apply_changes(network);
	}

	void StochasticGradientDescent::apply_momentum()
	{
		operations::modify(   d_weights.begin(),    d_weights.end(), [alpha = this->alpha] (Matrix& a) -> void { a *= alpha; });
		operations::modify(d_thresholds.begin(), d_thresholds.end(), [alpha = this->alpha] (Vector& a) -> void { a *= alpha; });
	}

	std::vector<double> StochasticGradientDescent::get_parameters() const
	{
		return std::vector<double>{eta, alpha};
	}

	/* ----- Adam ----- */

	Adam::Adam(const std::vector<size_t>& layout, const std::vector<double>& parameters)
		: Optimizer(layout, Method::ADAM),
			   weights_m(   d_weights.size()),    weights_v(   d_weights.size()),
			thresholds_m(d_thresholds.size()), thresholds_v(d_thresholds.size())
	{
		// Unpack parameters
		alpha = parameters[0];
		beta_1 = parameters[1];
		beta_2 = parameters[2];
		t = 0;

		// Initialize first and second moment collections
		operations::modify(weights_m.begin(), weights_m.end(), d_weights.begin(), [] (Matrix& m, const Matrix& d) -> void { m = Matrix(d.num_rows(), d.num_cols()); });
		operations::modify(weights_v.begin(), weights_v.end(), d_weights.begin(), [] (Matrix& m, const Matrix& d) -> void { m = Matrix(d.num_rows(), d.num_cols()); });

		operations::modify(thresholds_m.begin(), thresholds_m.end(), d_thresholds.begin(), [] (Vector& v, const Vector& d) -> void { v = Vector(d.length()); });
		operations::modify(thresholds_v.begin(), thresholds_v.end(), d_thresholds.begin(), [] (Vector& v, const Vector& d) -> void { v = Vector(d.length()); });
	}

	void Adam::optimize(NeuralNetwork& network, const Dataset& dataset)
	{
		// Increment time step
		t += 1.0;

		// Calculate gradient
		calculate_gradient(network, dataset);

		// Perform moment calculations
		for (size_t l = 0; l < d_weights.size(); l++)
			for (size_t i = 0; i < d_weights[l].num_rows(); i++)
				for (size_t j = 0; j < d_weights[l].num_cols(); j++)
					d_weights[l](i,j) = moment_calc(weights_g[l](i,j), weights_m[l](i,j), weights_v[l](i,j));

		for (size_t l = 0; l < d_thresholds.size(); l++)
			for (size_t i = 0; i < d_thresholds[l].length(); i++)
				d_thresholds[l][i] = moment_calc(thresholds_g[l][i], thresholds_m[l][i], thresholds_v[l][i]);

		// Apply changes
		apply_changes(network);
	}

	double Adam::moment_calc(double g, double& m, double& v)
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
		// Read in method
		int method;
		infile >> method;

		// Read in parameters
		int num_params;
		infile >> num_params;
		std::vector<double> params(num_params);
		for (double& p : params)
			infile >> p;

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

Vector Layer::compute() const
{
	return activator->activation(h());
}

Vector Layer::h() const
{
	return weights * nodes + thresholds;
}

Layer& Layer::operator=(const Vector& v)
{
	nodes = v;
	return *this;
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
	// Write layer parameters
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
	// Construct empty layer
	auto activator = activators::read_from_file(infile);
	size_t input_size, output_size;
	infile >> output_size >> input_size;

	Layer layer(input_size, output_size, std::move(activator));

	// Read in components
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
	layers.front() = inputs;

	// Feed output of each layer into the next
	operations::modify(layers.begin()+1, layers.end(), layers.begin(),
		[] (Layer& layer, const Layer& prev) -> void { layer = prev.compute(); });

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
	for (Layer& layer : layers)
		layer.randomize_weights(r);
}

double NeuralNetwork::loss(const Dataset& outputs, const Dataset& results)
{
	return losser->loss(outputs, results);
}

void NeuralNetwork::write_to_file(const std::string& filename)
{
	// Open file
	std::ofstream outfile(filename);

	// Have each component write itself
	outfile << layers.size() << std::endl;
	for (Layer& layer : layers)
		layer.write_to_file(outfile);

	losser->write_to_file(outfile);
	optimizer->write_to_file(outfile);

	// Close file
	outfile.close();
}

void NeuralNetwork::read_from_file(const std::string& filename)
{
	// Open file
	std::ifstream infile(filename);

	// Have each component read itself in
	int layer_count;
	infile >> layer_count;

	layers = std::vector<Layer>();
	layers.reserve(layer_count);
	for (size_t i = 0; i < layer_count; i++)
		layers.push_back(Layer::read_from_file(infile));

	losser = lossers::read_from_file(infile);
	optimizer = optimizers::read_from_file(infile, get_layout());

	// Close file
	infile.close();
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& network)
	: layers(network.layers)
{
	// Make pointers that can't be copied
	optimizer = optimizers::make_optimizer(network.optimizer->method, network.get_layout(), network.optimizer->get_parameters());
	losser = lossers::make_losser(network.losser->method);
}

NeuralNetwork& NeuralNetwork::operator=(const NeuralNetwork& network)
{
	layers = network.layers;

	// Make pointers that can't be copied
	optimizer = optimizers::make_optimizer(network.optimizer->method, network.get_layout(), network.optimizer->get_parameters());
	losser = lossers::make_losser(network.losser->method);
	return *this;
}

std::vector<size_t> NeuralNetwork::get_layout() const
{
	std::vector<size_t> layout(layers.size()+1);
	operations::modify(layout.begin(), layout.end(), layers.begin(), [] (size_t& a, const Layer& b) { a = b.nodes.length(); });
	layout.back() = layers.back().weights.num_rows();
	return layout;
}

