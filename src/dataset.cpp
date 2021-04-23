#include "dataset.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>

/* ----- Dataset ----- */

Dataset::Dataset()
{}

Dataset::Dataset(size_t a)
	: data(a)
{}

void Dataset::write_to_file(const std::string& filename) const
{
	std::ofstream outfile(filename);

	outfile << size() << std::endl;
	for (const point_type& p : data)
	{
		outfile << p.x.length();
		for (size_t i = 0; i < p.x.length(); i++)
			outfile << " " << p.x[i];

		outfile << p.y.length();
		for (size_t i = 0; i < p.y.length(); i++)
			outfile << " " << p.y[i];

		outfile << std::endl;
	}
}

Dataset Dataset::read_from_file(const std::string& filename)
{
	std::ifstream infile(filename);

	size_t datasize;
	infile >> datasize;

	Dataset dataset(datasize);

	for (point_type& p : dataset)
	{
		size_t s;

		infile >> s;
		p.x = Vector(s);
		for (size_t j = 0; j < s; j++)
			infile >> p.x[j];

		infile >> s;
		p.y = Vector(s);
		for (size_t j = 0; j < s; j++)
			infile >> p.y[j];
	}

	infile.close();

	return dataset;
}

Dataset Dataset::get_training_set(size_t index, size_t k) const
{
	Dataset training(size()-k);
	
	for (size_t i = 0; i < index; i++)
		training.data[i] = data[i];

	for (size_t i = index + k; i < size(); i++)
		training.data[i-k] = data[i];

	return training;
}

Dataset Dataset::get_validation_set(size_t index, size_t k) const
{
	Dataset validation(k);

	for (size_t i = 0; i < k; i++)
		validation.data[i] = data[index+i];

	return validation;
}

size_t Dataset::size() const
{
	return data.size();
}

/* ----- Dataset Helper Functions ----- */

Dataset generate_range(double low, double high, double step)
{
	Dataset data;
	for (double x = low; x < high; x += step)
	{
		Vector a(1);
		a[0] = x;
		data.data.push_back(Dataset::point_type(a, Vector(1)));
	}
	
	return data;
}


Dataset randomize_dataset_order(const Dataset& data)
{
	std::random_device rd;
	std::mt19937 g(rd());

	Dataset out = data;
	std::shuffle(out.begin(), out.end(), g);
	return out;
}

Dataset::container_type::iterator Dataset::begin()
{
	return data.begin();
}

Dataset::container_type::iterator Dataset::end()
{
	return data.end();
}

Dataset::container_type::const_iterator Dataset::begin() const
{
	return data.begin();
}

Dataset::container_type::const_iterator Dataset::end() const
{
	return data.end();
}

