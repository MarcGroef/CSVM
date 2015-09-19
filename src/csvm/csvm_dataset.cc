#include <csvm/csvm_dataset.h>

using namespace std;
using namespace csvm;


void CSVMDataset::loadCifar10(string dir){
   cifar10.loadImages(dir);
   
}