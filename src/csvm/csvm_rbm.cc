#include <csvm/csvm_rbm.h>

using namespace std;
using namespace csvm;

   union charInt{
      unsigned char chars[4];
      int intVal;
   };
   
   union charDouble{
      char chars[8];
      double doubleVal;
   };
   
   RBM::RBM(){
      settings.nLayers = 2;
      settings.layerSizes = (int*) malloc(2*sizeof(int));
      assert(settings.layerSizes!=NULL);
      settings.layerSizes[0] = 300;
      settings.layerSizes[1] = 100;
      settings.learningRate = 0.01;
      settings.nGibbsSteps = 1;
      
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
      
   }
   
   void RBM::setSettings(RBMSettings set){
      this->settings = set;
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
   
   void RBM::train(){
      cout << "Training rbm..\n";
      performRBM(&layers,&data,settings.learningRate,0,settings.nGibbsSteps);
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
   
   void RBM::exportNetwork(string filename){
   /* Format:
    * 4 bytes: int : number of layers
    * for each of layer: 4 bytes int layerSize;
    * 
    *
    */
   charDouble fancyDouble;
   charInt fancyInt;
   
   ofstream file(filename.c_str(), ios::out | ios::binary);
   
   
   //write number of layers
   fancyInt.intVal = settings.nLayers;
   for(size_t idx = 0; idx < 4; ++idx){
      file << fancyInt.chars[idx];      
   }
   //write layer sizes
   for(size_t layer = 0; layer < (unsigned int)settings.nLayers; ++layer){
      fancyInt.intVal = settings.layerSizes[layer];
      for(size_t idx = 0; idx < 4; ++idx){
         file << fancyInt.chars[idx];    
      }
   }
   //write weights 
   
   for(size_t layer = 0; layer < (unsigned int)settings.nLayers - 1; ++layer){
      for(size_t lIdx1 = 0; lIdx1 < (unsigned int) settings.layerSizes[layer]; ++lIdx1){  //source neuron index
         for(size_t lIdx2 = 0; lIdx2 < (unsigned int)settings.layerSizes[layer]; ++lIdx2){  //target neuron
            fancyDouble.doubleVal = layers.weights[layer][lIdx1][lIdx2];
            for(size_t idxt = 0; idxt < 8; ++ idxt)
               file << fancyDouble.chars[idxt];
         }
      }
   }
   
   file.close();
      
   }
