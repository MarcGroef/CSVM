# ifndef SVMENSEMBLE_H
# define SVMENSEMBLE_H

#include "SVM.h"
#include "utils.h"




class SVMEnsemble{

	string fnName;
	int cLev;	
	SVM* classSVMs;
	
	public:

	SVMEnsemble(int cLev, int nClasses, int nDims);
	~SVMEnsemble();

};


# endif
