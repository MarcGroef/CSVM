#ifndef CSVM_CLEAN_DESCRIPTOR_H
#define CSVM_CLEAN_DESCRIPTOR_H

#include <vector>
#include <cstdlib>


#include "csvm_patch.h"
#include "csvm_feature.h"

using namespace std;
namespace csvm{

   class CleanDescriptor{
   public:
      Feature describe(Patch p);
      
   };
   
   
   
}


#endif