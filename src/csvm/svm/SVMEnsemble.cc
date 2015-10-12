# include "SVMEnsemble.h"




SVMEnsemble::SVMEnsemble(int callLevel, int nClasses, int nDims): cLev(callLevel)
{
	fnName = "SVMEnsemble";
	if (verbose) report(cLev, fnName, "Constructing...");
	classSVMs = new SVM[nClasses];
	for (int i=0; i<nClasses; i++)	classSVMs[i] = SVM(cLev+1, nDims);

	if (verbose) report(cLev, fnName, "Done.");
}


SVMEnsemble::~SVMEnsemble()
{
	if (verbose) report(cLev, fnName, "Deleting...");
	delete [] classSVMs;
	if (verbose) report(cLev, fnName, "Done.");
}
