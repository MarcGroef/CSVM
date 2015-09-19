
#include <csvm/csvm.h>
#include <iostream>

using namespace csvm;
using namespace std;

int main(int argc,char**argv){
   CSVMClassifier c;
   c.dataset.loadCifar10("../datasets/cifar-10-batches-bin/data_batch_1.bin");
   return 0;
}






