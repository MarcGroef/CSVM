#include <csvm/csvm_rbm.h>

using namespace std;
using namespace csvm;
   
   RBM::RBM(){
      settings.nLayers = 2;
      settings.layerSizes = (int*) malloc(2*sizeof(int));
      assert(settings.layerSizes!=NULL);
      settings.layerSizes[0] = 5;
      settings.layerSizes[1] = 5;
      settings.learningRate = 0.01;
      
      initLayerStack(&layers,settings.nLayers,settings.layerSizes);
      initStackWeightsRandom(&layers);
   }
   
   RBM::RBM(int nLayers,int* layerSizes,float learningRate){
      initLayerStack(&layers,nLayers,layerSizes);
      this->settings.learningRate = learningRate;
      this->settings.layerSizes = layerSizes;
      this->settings.nLayers = nLayers;
   }
   
   
   RBM::RBM(int nLayers,int* layerSizes,float learningRate,double** dataset,int nDataEntries){
      initLayerStack(&layers,nLayers,layerSizes);
      data.data = dataset;
      data.size = nDataEntries;
      this->settings.learningRate = learningRate;
      this->settings.layerSizes = layerSizes;
      this->settings.nLayers = nLayers;
   }
   
   RBM::~RBM(){
      freeLayerStack(&layers);
      free(settings.layerSizes);
      //freeDataset();
   }
   
   void RBM::linkDataset(double** data,int nEntries){
      this->data.data = data; 
      this->data.size = nEntries;
     
   }
   
   void RBM::freeDataset(){
      unsigned int size = data.size;
      for(size_t idx = 0; idx < size; ++idx)
         free(data.data[idx]);
      free(data.data);
   }
   
   void RBM::train(float learningRate,int nGibbsSampleSteps){
      cout << "Training rbm..\n";
      performRBM(&layers,&data,learningRate,0,nGibbsSampleSteps);
   }
   
   double* RBM::getOutput(){
      double* output = new double[settings.layerSizes[settings.nLayers-1]];
      for (int i = 0; i < settings.layerSizes[settings.nLayers - 1];i++){
         output[i] = layers.layers[settings.nLayers-1][i];
      }
      return output;
   }
   
   double* RBM::run(double* input){
      for(int i = 0;i < settings.layerSizes[0];i++){
         layers.layers[0][i] = input[i];
      }
      flowUp(&layers);
      return getOutput();
   }
