#ifndef CSVM_CODEBOOK_H
#define CSVM_CODEBOOK_H

/* This class implements the Bag-of-Visual-Words model (a.k.a codebook)
 * It contains the centroids and can map a feature to an activation vector, given the centroids
 * 
 * 
 * 
 */



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
      SOFT_ASSIGNMENT_CLIPPING,
      COSINE_SOFT_ASSIGNMENT,
   };

   struct Codebook_settings{
      LVQ_Settings lvqSettings;
      KMeans_settings kmeansSettings;
	   AKMeans_settings akmeansSettings;
      CodebookClusterMethod method;
      unsigned int numberVisualWords;
      double similaritySigma;
      SimilarityFunction simFunction;
      bool generate;
   };

   
  
  class Codebook{
    Codebook_settings settings;
    LVQ lvq;
    KMeans kmeans;
	 AKMeans akmeans;

    vector<Centroid> bow;

    unsigned int nClasses;
    void standardize(vector<double>& x);
  public:
    bool debugOut, normalOut;
    Codebook();
    void constructCodebook(vector<Feature> featureset);
    void setSettings(Codebook_settings s);
    Centroid getCentroid(int centrIdx);
    vector < double > getActivations(vector<Feature> features);
    void exportCodebook(string filename);
    void importCodebook(string filename);
    unsigned int getNClasses();
    unsigned int getNCentroids();
    void constructActivationCodebook(vector<Feature> activations, unsigned int layerIdx);
    vector< double > getQActivations(vector<Feature> features);
    bool getGenerate();
    
	//for akmeans:
	//vector<vector< double> > Codebook::getAKContributions(vector<Feature> classifyFeatures);
	vector<vector <double> > getCentroidByClassContributions();
	vector<double> getCentroidByClassContributions(int cl);
	vector<double> getCentroidByClassContributions(Feature feat);
  };
  
}

#endif
