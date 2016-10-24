#ifndef CSVM_LINEAR_NETWORK
#define CSVM_LINEAR_NETWORK

/* A linear network - classifier class. Easilly converted to a one-layer neural net by using sigmoids
 * 
 * 
 * 
 * 
 */

#include <csvm/csvm_dataset.h>
#include <vector>
#include <cmath>
using namespace std;

namespace csvm{
   
   struct LinNetSettings{
      float initWeight;
      float learningRate;
      bool useSigmoid;
      bool useDifferentCodebooksPerClass;
      unsigned int nClasses;
      unsigned int nCentroids;
      unsigned int nIter;
      
   };
 
   class LinNetwork{
      float initWeights;
      vector< vector<float> > weights;
      vector< float > biases;
      float computeOutput(unsigned int networkClassIdx, vector<float>& clActivations);
      
      LinNetSettings settings;
   public:
      bool debugOut, normalOut;
      LinNetwork();//(unsigned int nClasses,unsigned int nCentroids, float initWeights);
      void setSettings(LinNetSettings s);
      void train(vector< vector< float > >& activations, CSVMDataset* ds);
      unsigned int classify(vector< float >imageActivations);
   };
   
}

#endif
