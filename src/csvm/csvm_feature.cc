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

void Feature::setLabelId(int id){
   labelId = id;
}

int Feature::getLabelId(){
   return labelId;
}
double Feature::getDistanceSq(Feature* f){
   //cout << "My size = " << size << ", the other one's = " << f->size << endl;
   if(f->size != size){
      cout << "csvm::Feature::getDistance() Error! Different feature sizes!\n";
      exit(-1);
   }
   double distance = 0;
   double dist;
   for(int dim = 0; dim < size; ++dim){
      dist = (f->content[dim] - content[dim]);
      //if(dim==0)cout << "delta: " << dist << " between " << f->content[dim] << " and " << content[dim] << endl;
      distance += dist * dist;
   }
   //cout << "dist = " << distance << endl;
   return distance;
}

double Feature::getManhDist(Feature* f){
   if(f->size != size){
      cout << "csvm::Feature::getDistance() Error! Different feature sizes!\n";
      exit(-1);
   }
   double distance = 0;
   double dist;
   for(int dim = 0; dim < size; ++dim){
      dist = (f->content[dim] - content[dim]);
      distance += dist >= 0 ? dist : dist*-1 ;
   }
   return distance;
}