#include <csvm/csvm_rbm.h>

using namespace std;
using namespace csvm{
   
   RBM::RBM(int nLayers,int* layerSizes,float learningRate){
      initLayerStack(&layers,nLayers,layerSizes);
      this->learningRate = learningRate;
     
   }
   
   
   RBM::RBM(int nLayers,int* layerSizes,float learningRate,double** dataset,int nDataEntries){
      initLayerStack(&layers,nLayers,layerSizes);
      dataset->data = dataset;
      dataset->size = nDataEntries;
      this->learningRate = learningRate;
   }
   
   RBM::~RBM(){
      freeLayerStack(&layers);
      
   }
      
      
}