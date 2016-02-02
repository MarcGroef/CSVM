 #include <csvm/csvm_feature.h>
 
/* This Feature class is a bit like a vector, except that it also contains a label.
 * Also, it has some often used distance functions with it 
 * 
 * 
 */


using namespace std;
using namespace csvm;

Feature::Feature(int size,double initValue){
   content = vector<double>(size,initValue);
   this->size = size;
}

Feature::Feature(Feature* f){
   content = vector<double>(f->content);
   this->size = f->size;
   this->label = f->label;
   
}

Feature::Feature(vector<double>& vect){
   content = vect;
   size = vect.size();
   this->label = "hoi";
}

void Feature::setLabelId(int id){
   labelId = id;
}

unsigned int Feature::getLabelId(){
   return labelId;
}

//get squared distance

double Feature::getDistanceSq(Feature& f){
   //cout << "My size = " << size << ", the other one's = " << f->size << endl;
   if(f.size != size){
      cout << "csvm::Feature::getDistance() Error! Different feature sizes! Namely " << f.size << " vs. " << size << endl;
      exit(-1);
   }
   //if(f->content == this->content) cout << "Same pointer also!!!\n";
   double distance = 0;
   double dist;
   for(int dim = 0; dim < size; ++dim){
      dist = (f.content[dim] - content[dim]);
      //if(dist == 0.f) cout << "exactly the same element! Namely " << f.content[dim] << " and " << content[dim] << "\n";
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