#ifndef DATASET_H
#define DATASET_H

#include <string>
#include "linear.h"

/*
 * A dataset containing paired input and output points (Vectors).
 * Input data is labelled as x.
 * Output data is labelled as y.
 */
class Dataset
{
public:
	/*
	 * Construct an empty dataset.
	 */
	Dataset();

	/*
	 * Construct a dataset of a elements, all of which are zero.
	 */
	Dataset(size_t a);

	/*
	 * Print the dataset.
	 */
	void print() const;

	/*
	 * Number of points in the dataset.
	 */
	size_t size() const;

	/*
	 * Read and write dataset to a file.
	 */
	void write_to_file(const std::string& filename) const;
	static Dataset read_from_file(const std::string& filename);	

	/*
	 * Get the ith training / validation set for k-fold cross validation
	 */
	Dataset get_training_set(size_t index, size_t subset_size) const;
	Dataset get_validation_set(size_t index, size_t subset_size) const;
	
	/*
	 * Dataset input (x) and output (y)
	 */
	std::vector<Vector> x;
	std::vector<Vector> y;
};

/*
 * Generates a range of doubles for the specified parameters.
 */
std::vector<Vector> generate_range(double low, double high, double step);

/*
 * Randomizes the order of the given dataset.
 */
Dataset randomize_dataset_order(const Dataset& data);

#endif
