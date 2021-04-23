#include "linear.h"

/* ----- Printing Functions ----- */

void print_vec(const Vector& v)
{
	std::cout << "[";
	if (v.length() > 0)
		std::cout << v[0];
	
	for (size_t i = 1; i < v.length(); i++)
		std::cout << ", " << v[i];
	
	std::cout << "]" << std::endl;
}

void print_matrix(const Matrix& m)
{
	for (size_t i = 0; i < m.num_rows(); i++)
	{
		std::cout << "[";
		if (m.num_cols() > 0)
			std::cout << m(i,0);
		
		for (size_t j = 1; j < m.num_cols(); j++)
			std::cout << ", " << m(i,j);
		
		std::cout << "]" << std::endl;
	}
}

/* ----- Vector ----- */

Vector::Vector()
{}

Vector::Vector(size_t size)
	: data(size, 0)
{}

double Vector::operator[](size_t i) const
{
	return data[i];
}

double& Vector::operator[](size_t i)
{
	return data[i];
}

Vector& Vector::operator+=(const Vector& v) {
	component_wise(v, operations::plus_equals<double>());
	return *this;
}

Vector& Vector::operator-=(const Vector& v) {
	component_wise(v, operations::minus_equals<double>());
	return *this;
}

Vector& Vector::operator*=(const Vector& v) {
	component_wise(v, operations::times_equals<double, double>());
	return *this;
}

Vector& Vector::operator*=(double t) {
	component_wise(t, operations::times_equals<double, double>());
	return *this;
}

Vector& Vector::operator/=(double t) {
	component_wise(t, operations::divides_equals<double, double>());
	return *this;
}

void Vector::clear()
{
	std::fill(data.begin(), data.end(), 0);
}

size_t Vector::length() const
{
	return data.size();
}

/* ----- Matrix ----- */


Matrix::Matrix()
{}

Matrix::Matrix(size_t m, size_t n)
	: data(m, Vector(n))
{}

size_t Matrix::num_rows() const
{
	return data.size();
}

size_t Matrix::num_cols() const
{
	return data[0].length();
}

double Matrix::operator()(size_t i, size_t j) const
{
	return data[i][j];
}

double& Matrix::operator()(size_t i, size_t j)
{
	return data[i][j];
}

Matrix& Matrix::operator+=(const Matrix& matrix)
{
	component_wise(matrix, operations::plus_equals<Vector>());
	return *this;
}

Matrix& Matrix::operator-=(const Matrix& matrix)
{
	component_wise(matrix, operations::minus_equals<Vector>());
	return *this;
}

Matrix& Matrix::operator*=(double t) {
	component_wise(t, operations::times_equals<Vector, double>());
	return *this;
}

void Matrix::clear() {
	std::for_each(data.begin(), data.end(), [] (Vector& v) { v.clear(); });
}

/* ----- Vector & Matrix friend Operations ----- */

Vector operator+(Vector a, const Vector& b) {
	a += b;
	return a;
}

Vector operator-(Vector a, const Vector& b) {
	a -= b;
	return a;
}

Vector operator*(Vector a, const Vector& b) {
	a *= b;
	return a;
}

Vector operator*(Vector a, double t) {
	a *= t;
	return a;
}

Vector operator/(Vector a, double t) {
	a /= t;
	return a;
}

Vector operator*(const Matrix& m, const Vector& v)
{
	Vector out(m.num_rows());

	std::transform(m.data.begin(), m.data.end(), out.data.begin(),
		[v] (const Vector& a) -> double {
			return dot(a,v);
		}
	);

	return out;
}

Vector operator*(const Vector& v, const Matrix& m)
{
	Vector out(m.num_cols());
	for (size_t i = 0; i < m.num_cols(); i++)
		for (size_t j = 0; j < m.num_rows(); j++)
			out[i] += v[j] * m(j,i);
	return out;
}

Matrix operator*(double a, const Matrix& m)
{
	Matrix out = m;
	out *= a;
	return out;
}

Vector operator*(double a, const Vector& v)
{
	Vector out = v;
	out *= a;
	return out;
}

double dot(const Vector& a, const Vector& b)
{
	return std::inner_product(a.data.begin(), a.data.end(), b.data.begin(), 0.0);
}
