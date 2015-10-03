 #include <csvm/csvm_feature.h>



using namespace std;
using namespace csvm;

Feature::Feature(int size,double initValue){
   content = vector<double>(size,initValue);
}

