# include "SVM.h"


SVM::SVM(void)
{	// Dummi constructor for array initialization
	fnName = "SVM_PLACEHOLDER";
}

	
SVM::SVM(int callLevel, int nDimensions): cLev(callLevel), nDims(nDimensions)
{
	fnName = "SVM";
	
	bias = 0.0;
	weights = new double[nDims];

	if (verbose) report(cLev, fnName, "Constructed.");
}


SVM::~SVM(){
	if (verbose) report(cLev, fnName, "Deleted.");
}
