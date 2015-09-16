#include <csvm/csvm_rbm.h>

using namespace std;
using namespace csvm;

   
   RBM::RBM(int nLayers,int* layerSizes,float learningRate){
      initLayerStack(&layers,nLayers,layerSizes);
      this->learningRate = learningRate;
      this->layerSizes = layerSizes;
      this-> nLayers = nLayers;
   }
   
   
   RBM::RBM(int nLayers,int* layerSizes,float learningRate,double** dataset,int nDataEntries){
      initLayerStack(&layers,nLayers,layerSizes);
      data.data = dataset;
      data.size = nDataEntries;
      this->learningRate = learningRate;
      this->layerSizes = layerSizes;
      this->nLayers = nLayers;
   }
   
   RBM::~RBM(){
      freeLayerStack(&layers);
      
   }
   
   void RBM::linkDataset(double** data,int nEntries){
      this->data.data = data; 
      this->data.size = nEntries;
     
   }
   
   void RBM::train(float learningRate,int nGibbsSampleSteps){
      performRBM(&layers,&data,learningRate,0,nGibbsSampleSteps);
   }
   
   double* RBM::getOutput(){
      double* output = new double[layerSizes[layerSizes[nLayers-1]]];
	  for (int i = 0; i < layerSizes[nLayers - 1];i++){
         output[i] = layers.layers[nLayers-1][i];
      }
      return output;
   }
   
   double* RBM::run(double* input){
      for(int i = 0;i < layerSizes[0];i++){
         layers.layers[0][i] = input[i];
      }
      flowUp(&layers);
      return getOutput();
   }
