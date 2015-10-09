#ifndef CSVM_CODEBOOK_H
#define CSVM_CODEBOOK_H

#include <cmath>
#include <iostream>
#include <fstream>

#include "csvm_feature.h"
#include "csvm_lvq.h"
#include "csvm_kmeans.h"

using namespace std;
namespace csvm{
  
   enum CodebookClusterMethod{
   LVQ_Clustering = 0,
   KMeans_Clustering = 1,

   };

   struct Codebook_settings{
   LVQ_Settings lvqSettings;
   KMeans_settings kmeansSettings;
   CodebookClusterMethod method;
   unsigned int numberVisualWords;
   };

   
  
  class Codebook{
    Codebook_settings settings;
    LVQ lvq;
    KMeans kmeans;
    vector< vector<Feature> > bow;
    unsigned int nClasses;
  public:
    Codebook();
    void constructCodebook(vector<Feature> featureset,int labelId);
    vector<Feature> getActivations(vector<Feature> features);
    void exportCodebook(string filename);
    void importCodebook(string filename);
  };
  
}

#endif