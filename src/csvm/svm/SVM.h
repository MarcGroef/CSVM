# ifndef SVM_H 
# define SVM_H

# include "utils.h"


class SVM{

	string fnName;
	int cLev;

	double bias;
	double* weights;
	int nDims;

	public:

	SVM();
	SVM(int callLevel, int nDimensions);
	~SVM();

};


#endif
