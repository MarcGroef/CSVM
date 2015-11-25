#ifndef CSVM_LINEAR_NETWORK
#define CSVM_LINEAR_NETWORK

#include <csvm/csvm_dataset.h>
#include <vector>

using namespace std;

namespace csvm{
   
 
   class LinNetwork{
      unsigned int nClasses;
      unsigned int nCentroids;
      double initWeights;
      vector< vector< vector<double> > >weights;
     
      double computeOutput(unsigned int networkClassIdx, vector< vector<double> >& clActivations);
   public:
      LinNetwork(unsigned int nClasses,unsigned int nCentroids, double initWeights);
      void train(vector< vector< vector< double > > >& clActivations, CSVMDataset* ds);
      unsigned int classify(vector< vector< double > >& imageActivations);
   };
   
}

#endif