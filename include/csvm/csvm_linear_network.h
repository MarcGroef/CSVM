#ifndef CSVM_LINEAR_NETWORK
#define CSVM_LINEAR_NETWORK

#include <csvm/csvm_dataset.h>
#include <vector>
#include <cmath>
using namespace std;

namespace csvm{
   
   struct LinNetSettings{
      double initWeight;
      double learningRate;
      bool useSigmoid;
      bool useDifferentCodebooksPerClass;
      unsigned int nClasses;
      unsigned int nCentroids;
      unsigned int nIter;
      
   };
 
   class LinNetwork{
      double initWeights;
      vector< vector<double> > weights;
      vector< double > biases;
      double computeOutput(unsigned int networkClassIdx, vector<double>& clActivations);
      
      LinNetSettings settings;
   public:
      bool debugOut, normalOut;
      LinNetwork();//(unsigned int nClasses,unsigned int nCentroids, double initWeights);
      void setSettings(LinNetSettings s);
      void train(vector< vector< double > >& activations, CSVMDataset* ds);
      unsigned int classify(vector< double >imageActivations);
   };
   
}

#endif
