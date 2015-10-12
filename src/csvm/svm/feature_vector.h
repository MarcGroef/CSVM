# ifndef FEATURE_VECTOR_H
# define FEATURE_VECTOR_H

# include "utils.h"


class feature_vector{

	string fnName;							// Both for logging (see utils.h)
	int cLev;

	public:

	int nDims;								// Number of dimensions
	float* values;							// The activation values for different features 
	feature_vector(int dimensions);
	feature_vector(int callLevel, int dimensions);
	~feature_vector();

};


#endif

