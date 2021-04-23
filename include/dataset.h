#ifndef DATASET_H
#define DATASET_H

#include <string>
#include "linear.h"

/*
 * A point representing a data pair.
 */
template<typename T> 
struct Point
{
	Point()
		: x(), y()
	{}

	Point(const T& a, const T& b)
		: x(a), y(b)
	{}

	T x;
	T y;
};

/*
 * A dataset containing paired input and output points (Vectors).
 * Input data is labelled as x.
 * Output data is labelled as y.
 */
class Dataset
{
public:
	using point_type = Point<Vector>;
	using container_type = std::vector<point_type>;

	/*
	 * Construct an empty dataset.
	 */
	Dataset();

	/*
	 * Construct a dataset of a elements, all of which are zero.
	 */
	Dataset(size_t a);

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
	 * Iterators to underlying container
	 */
	container_type::iterator begin();
	container_type::iterator   end();

	container_type::const_iterator begin() const;
	container_type::const_iterator end()   const;

	/*
	 * Dataset of points
	 */
	container_type data;
};

/*
 * Generates a range of doubles for the specified parameters.
 */
Dataset generate_range(double low, double high, double step);

/*
 * Randomizes the order of the given dataset.
 */
Dataset randomize_dataset_order(const Dataset& data);

#endif
