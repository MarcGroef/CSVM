#ifndef CSVM_RBM_H
#define CSVM_RBM_H

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
      
      
      
      RBM();
      RBM(int nLayers,int* layerSizes,float learningRate);
      RBM(int nLayers,int* layerSizes,float learningRate,double** dataset,int nDataEntries);  //dimension of a single data-entry should be equal to the layersize of the first, thus zero-th, layer.
      ~RBM();
      
      void setSettings(RBMSettings set);
      void freeDataset();
      void linkDataset(double** data,int nEntries);
      void train();
      double* getOutput();
      double* run(double* input);
      void exportNetwork(string filename);
   };
   
   
}

#endif