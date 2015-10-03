#ifndef CSVM_RBM_H
#define CSVM_RBM_H

extern "C"{
#include <dnn/dnn.h>
}
#include <vector>

using namespace std;




namespace csvm{
   
   struct RBMSettings{
      
      
   };
   
   
   
   class RBM{
      LayerStack layers;
      float learningRate;
      Dataset data;
      int* layerSizes;
      int nLayers;
      
     
   public:
      
      
      
      
      RBM(int nLayers,int* layerSizes,float learningRate);
      RBM(int nLayers,int* layerSizes,float learningRate,double** dataset,int nDataEntries);  //dimension of a single data-entry should be equal to the layersize of the first, thus zero-th, layer.
      ~RBM();
      
      void linkDataset(double** data,int nEntries);
      void train(float learningRate,int nGibbsSampleSteps);
      double* getOutput();
      double* run(double* input);
   };
   
   
}

#endif