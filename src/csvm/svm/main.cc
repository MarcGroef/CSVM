// Attempt at a new SVM Code compatible with feature vectors etc.

# include "main.h"

using namespace std;


bool verbose = true;



// Main: purely to get things going...
int main(int argn, char *argv[])
{
	string fnName = "Main";
	int cLev = 0;
	if (verbose) report(cLev, fnName, "Initializing program...");
 	int nClasses    = 2;		// Number of classes
	int nDims   	= 100;		// Dimensionality of the feature Vectors

	if (verbose) report(cLev, fnName, "Initializing Class-specific SVMs:");
   	SVMEnsemble svms(cLev+1, nClasses, nDims);
}





















