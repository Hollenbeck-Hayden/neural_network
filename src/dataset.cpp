#include "dataset.h"

#include <iostream>
#include <fstream>
#include "randomizer.h"

/* ----- Dataset ----- */

Dataset::Dataset()
{}

Dataset::Dataset(size_t a)
	: x(a), y(a)
{}

void Dataset::print() const
{
	std::cout << "x:\n";
	for (size_t i = 0; i < x.size(); i++)
		print_vec(x[i]);

	std::cout << "y: ";
	for (size_t j = 0; j < y.size(); j++)
		print_vec(y[j]);
}

void Dataset::write_to_file(const std::string& filename) const
{
	std::ofstream outfile(filename);

	outfile << size() << std::endl;
	for (size_t i = 0; i < x.size(); i++)
	{
		outfile << x[i].length();
		for (size_t j = 0; j < x[i].length(); j++)
			outfile << " " << x[i][j];
		outfile << std::endl;
	}

	for (size_t i = 0; i < y.size(); i++)
	{
		outfile << y[i].length();
		for (size_t j = 0; j < y[i].length(); j++)
			outfile << " " << y[i][j];
		outfile << std::endl;
	}
}

Dataset Dataset::read_from_file(const std::string& filename)
{
	std::ifstream infile(filename);

	size_t datasize;
	infile >> datasize;

	Dataset dataset(datasize);

	for (size_t i = 0; i < dataset.x.size(); i++)
	{
		size_t s;
		infile >> s;
		dataset.x[i] = Vector(s);
		for (size_t j = 0; j < dataset.x[i].length(); j++)
			infile >> dataset.x[i][j];
	}

	for (size_t i = 0; i < dataset.y.size(); i++)
	{
		size_t s;
		infile >> s;
		dataset.y[i] = Vector(s);
		for (size_t j = 0; j < dataset.y[i].length(); j++)
			infile >> dataset.y[i][j];
	}

	infile.close();

	return dataset;
}

Dataset Dataset::get_training_set(size_t index, size_t k) const
{
	Dataset training(size()-k);
	
	for (size_t i = 0; i < index; i++)
	{
		training.x[i] = x[i];
		training.y[i] = y[i];
	}

	for (size_t i = index + k; i < size(); i++)
	{
		training.x[i-k] = x[i];
		training.y[i-k] = y[i];
	}

	return training;
}

Dataset Dataset::get_validation_set(size_t index, size_t k) const
{
	Dataset validation(k);

	for (size_t i = 0; i < k; i++)
	{
		validation.x[i] = x[index+i];
		validation.y[i] = y[index+i];
	}

	return validation;
}

size_t Dataset::size() const
{
	return x.size();
}

/* ----- Dataset Helper Functions ----- */

std::vector<Vector> generate_range(double low, double high, double step)
{
	std::vector<Vector> data;

	for (double x = low; x < high; x += step)
	{
		Vector a(1);
		a[0] = x;
		data.push_back(a);
	}
	
	return data;
}


Dataset randomize_dataset_order(const Dataset& data)
{
	Randomizer r(0, 1);
	Dataset temp = data;
	Dataset out;

	while (temp.size() > 0)
	{
		int index = (int)( ((double) temp.size()) * r.next() );
		out.x.push_back(temp.x[index]);
		out.y.push_back(temp.y[index]);
		temp.x.erase(temp.x.begin() + index);
		temp.y.erase(temp.y.begin() + index);
	}

	return out;
}






