#ifndef LINEAR_H
#define LINEAR_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>

/* ----- Forward Declarations ----- */

class Vector;
class Matrix;

/* ----- Printing Methods ----- */

void print_vec(const Vector& v);
void print_matrix(const Matrix& m);

/*
 * Prints std::vector to the console.
 */
template<typename T>
void print_vec(const std::vector<T>& v)
{
	std::cout << "[";
	if (v.size() > 0)
		std::cout << v[0];
	
	for (size_t i = 1; i < v.size(); i++)
		std::cout << ", " << v[i];
	
	std::cout << "]" << std::endl;
}

/* ----- Vector ----- */

/*
 * Variable size (math) vector class.
 */
class Vector
{
public:
	/*
	 * Construct an empty vector.
	 */
	Vector();

	/*
	 * Construct a vector filled with 0 of length size.
	 */
	Vector(size_t size);

	/*
	 * No implicit conversions between matrices and vectors.
	 */
	Vector(const Matrix&) = delete;
	Vector(Matrix&&) = delete;
	Vector& operator=(const Matrix&) = delete;
	Vector& operator=(Matrix&&) = delete;

	/*
	 * Access operators
	 */
	double operator[](size_t i) const;
	double& operator[](size_t i);

	/*
	 * Set vector components to 0.
	 */
	void clear();

	/*
	 * Return the length of the vector.
	 */
	size_t length() const;

	/* ----- Arithmetic Operations ----- */
	Vector& operator+=(const Vector& v);
	Vector& operator-=(const Vector& v);
	Vector& operator*=(const Vector& v);
	Vector& operator*=(double t);
	Vector& operator/=(double t);

	friend Vector operator+(Vector a, const Vector& b);
	friend Vector operator-(Vector a, const Vector& b);
	friend Vector operator*(Vector a, const Vector& b);
	friend Vector operator*(Vector a, double t);
	friend Vector operator/(Vector a, double t);

	friend Vector operator*(const Matrix& m, const Vector& v);
	friend Vector operator*(double t, const Vector& v);
	friend Vector operator*(const Vector& v, const Matrix& m);

private:
	std::vector<double> data;
};

/* ----- Matrix ----- */

/*
 * Variable size matrix.
 */
class Matrix
{
public:
	/*
	 * Construct an empty matrix.
	 */
	Matrix();

	/*
	 * Construct a MxN matrix filled with 0.
	 */
	Matrix(size_t m, size_t n);

	/*
	 * No implicit conversion to Vector.
	 */
	Matrix(const Vector&) = delete;
	Matrix(Vector&&) = delete;
	Matrix& operator=(const Vector&) = delete;
	Matrix& operator=(Vector&&) = delete;
	
	/*
	 * Dimensions of the matrix.
	 */
	size_t num_rows() const;
	size_t num_cols() const;

	/*
	 * Access operators.
	 */
	double operator()(size_t i, size_t j) const;
	double& operator()(size_t i, size_t j);

	/*
	 * Set all components to 0.
	 */
	void clear();

	/* ----- Arithmetic operations ----- */
	Matrix& operator+=(const Matrix& matrix);
	Matrix& operator-=(const Matrix& matrix);
	Matrix& operator*=(double t);

	friend Vector operator*(const Matrix& m, const Vector& v);
	friend Matrix operator*(double t, const Matrix& m);
	friend Vector operator*(const Vector& v, const Matrix& m);

private:
	std::vector<Vector> data;
};

#endif
