#ifndef RANDOMIZER_H
#define RANDOMIZER_H

#include <random>

/*
 * A class which handles generating random variables.
 */
class Randomizer
{
public:
	/*
	 * Construct a randomizer which generates random numbers
	 * from low to high.
	 */
	Randomizer(double low, double high)
	{
		engine = std::default_random_engine(r());
		urand = std::uniform_real_distribution<double>(low, high);
	}

	/*
	 * Get the next random number.
	 */
	double next()
	{
		return urand(engine);
	}

private:
	std::random_device r;
	std::default_random_engine engine;
	std::uniform_real_distribution<double> urand;
};

#endif
