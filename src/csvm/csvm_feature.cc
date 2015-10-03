 #include <csvm/csvm_feature.h>



using namespace std;
using namespace csvm;

Feature::Feature(int size,double initValue){
   content = vector<double>(size,initValue);
   this->size = size;
}

Feature::Feature(Feature* f){
   content = f->content;
   this->size = f->size;
   this->label = f->label;
   
}