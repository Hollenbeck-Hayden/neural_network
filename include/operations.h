#ifndef OPERATIONS_H
#define OPERATIONS_H

/*
 * A set of algorithms and operations that are used throughout the program.
 */
namespace operations
{
	/* ----- In-place Transformation ----- */

	/*
	 * Modifies the input container in place.
	 * BinaryOp should take a reference to the first argument (input to be modified), and
	 * expect the modifier to be provided as the second argument.
	 */
	template<class InputIt, class ModifierIt, class BinaryOp>
	void modify(InputIt input_first, InputIt input_last, ModifierIt modifier_first, BinaryOp op)
	{
		while (input_first != input_last)
			op(*input_first++, *modifier_first++);
	}

	/*
	 * Modifies the input container in place.
	 * The unary operation is performed on each element, where the argument should be
	 * taken as a reference.
	 */
	template<class InputIt, class UnaryOp>
	void modify(InputIt first, InputIt last, UnaryOp op)
	{
		while (first != last)
			op(*first++);
	}

	/* ----- Operation-Assignment Function Classes ----- */

	template<class T>
	class plus_equals
	{
	public:
		constexpr void operator()(T& a, const T& b) const { a += b;};
	};

	template<class T>
	class minus_equals
	{
	public:
		constexpr void operator()(T& a, const T& b) const { a -= b;};
	};

	template<class T, class V>
	class times_equals
	{
	public:
		constexpr void operator()(T& a, const V& b) const { a *= b;};
	};

	template<class T, class V>
	class divides_equals
	{
	public:
		constexpr void operator()(T& a, const V& b) const { a /= b;};
	};


}


#endif
