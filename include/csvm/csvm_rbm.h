#ifndef CSVM_RBM_H
#define CSVM_RBM_H

//DEPRECATED

#include <cstdlib>
#include <assert.h>
#include <iostream>
#include <fstream>

extern "C"{
#include <dnn/dnn.h>
}
#include <vector>

using namespace std;




namespace csvm{
   
   typedef struct{
      int nLayers;
      int* layerSizes;   //freed b rbm.cc RBM destructor
      float learningRate;
      int nGibbsSteps;
   }RBMSettings;
   
   
   
   class RBM{
      LayerStack layers;
      Dataset data;
      RBMSettings settings;
     
   public:
      bool debugOut, normalOut;
      
      
      RBM();
      RBM(int nLayers,int* layerSizes,float learningRate);
      RBM(int nLayers,int* layerSizes,float learningRate,float** dataset,int nDataEntries);  //dimension of a single data-entry should be equal to the layersize of the first, thus zero-th, layer.
      ~RBM();
      
      void setSettings(RBMSettings set);
      void freeDataset();
      void linkDataset(float** data,int nEntries);
      void train();
      float* getOutput();
      float* run(float* input);
      void exportNetwork(string filename);
      void importNetwork(string filename);
   };
   
   
}

#endif