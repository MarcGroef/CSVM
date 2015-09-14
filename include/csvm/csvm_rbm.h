#ifndef CSVM_RBM_H
#define CSVM_RBM_H

extern "C"{
#include <dnn/dnn.h>
}
#include <vector>

using namespace std;




namespace csmv{
   
   class RBM{
      LayerStack layers;
      float learningRate;
      
   public:
      Dataset data;
      
      
      
      RBM(int nLayers,int* layerSizes,float learningRate);
      RBM(int nLayers,int* layerSizes,float learningRate,double** dataset,int nDataEntries);  //dimension of a single data-entry should be equal to the layersize of the first, thus zero-th, layer.
      ~RBM();
      
   }
   
   
}

#endif