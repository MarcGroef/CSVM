#ifndef CSVM_CODEBOOK_H
#define CSVM_CODEBOOK_H

#include <cmath>
#include <iostream>
#include <fstream>

#include "csvm_feature.h"
#include "csvm_lvq.h"
#include "csvm_kmeans.h"
#include "csvm_annotated_kmeans.h"
#include "csvm_centroid.h"

using namespace std;
namespace csvm{
  
   enum CodebookClusterMethod{
      LVQ_Clustering = 0,
      KMeans_Clustering = 1,
	   AKMeans_Clustering = 2,
   };
   
   enum SimilarityFunction{
      CB_RBF,
      SOFT_ASSIGNMENT,
   };

   struct Codebook_settings{
      LVQ_Settings lvqSettings;
      KMeans_settings kmeansSettings;
	  AKMeans_settings akmeansSettings;
      CodebookClusterMethod method;
      unsigned int numberVisualWords;
      double similaritySigma;
      SimilarityFunction simFunction;
      bool useDifferentCodebooksPerClass;
      bool standardizeActivations;
   };

   
  
  class Codebook{
    Codebook_settings settings;
    LVQ lvq;
    KMeans kmeans;
	AKMeans akmeans;

    vector< vector<Centroid> > bow;

    unsigned int nClasses;
  public:
    Codebook();
    void constructCodebook(vector<Feature> featureset,int labelId);
    void setSettings(Codebook_settings s);
    Centroid getCentroid(int cl, int centrIdx);
    vector<vector < double > > getActivations(vector<Feature> features);
    void exportCodebook(string filename);
    void importCodebook(string filename);
    unsigned int getNClasses();
    unsigned int getNCentroids();
    void constructActivationCodebook(vector<Feature> activations, unsigned int layerIdx);

	//for akmeans:
	vector<vector <double> > getCentroidByClassContributions();
	vector<double> getCentroidByClassContributions(int cl);
  };
  
}

#endif