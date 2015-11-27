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
      bool useLinNet;
   };
 
   class LinNetwork{
      unsigned int nClasses;
      unsigned int nCentroids;
      double initWeights;
      vector< vector< vector<double> > >weights;
      vector< double > biases;
      double computeOutput(unsigned int networkClassIdx, vector< vector<double> >& clActivations);
      
      LinNetSettings settings;
   public:
      LinNetwork();//(unsigned int nClasses,unsigned int nCentroids, double initWeights);
      void setSettings(LinNetSettings s);
      void train(vector< vector< vector< double > > >& activations, CSVMDataset* ds);
      unsigned int classify(vector< vector< double > > imageActivations);
   };
   
}

#endif