#include <csvm/csvm_hog_descriptor.h>

using namespace cv;
using namespace std;
using namespace csvm;

HOGDescriptor::HOGDescriptor(int nBins=9,int cellSize=3,int cellStride=3,int blockSize=36,int blockStride=18){
   this->nBins=nBins;
   this->cellSize=cellSize;
   this->cellStride = cellStride;
   this->blockSize = blockSize;
   this->blockStride = blockStride;
}

//assumes a square uchar (grey) image


//Devnote: This function implements classic HOG, including how to partitionize the image. CSVM will do this in another way, so it's not quite finished
vector< vector<float> > HOGDescriptor::getHOG(Mat image){
   Mat gx,gy;
   gx = image.clone();  //cloned image to store horizontal gradient
   gy = image.clone();  //cloned image to store vertical gradient
   
   int imWidth = image.cols;
   int imHeight = image.rows;
   
   vector< vector<float> > histograms;   //collection of HOG histograms
   double* votes = new double[nBins]();
   double binSize = M_PI/nBins;
   
   if(image.cols != image.rows)
      cout << "HOGDescriptor::getHOG() Warning! Image is not a square! Some parts might not get scanned.\n";
   
   //get gradients of image
   for(int i=0; i<imWidth; i++)
      for(int j=0;j < imHeight; j++){
         int gv=(i -1 >= 0 ? image.at<uchar>(j, i - 1) : 0) - (i + 1 < imWidth ? image.at<uchar>(j, i + 1): 0);    //calculate difference between left and right pixel
         gx.at<uchar>(j,i) = (uchar)abs(gv);                                                    //store absolute difference
         gv = (j - 1 >= 0 ? image.at<uchar>(j - 1, i) : 0)-(j+1<imHeight?image.at<uchar>(j + 1, i) : 0);       //same for up and down
         gy.at<uchar>(j,i) = (uchar)abs(gv);
      }
   
   
   //show the gradients for fun
   
   namedWindow("Gx", WINDOW_AUTOSIZE);   //spawn window called "Gx"
   imshow("Gx", gx);                     //parse the image to the just spawned "Gx"
   waitKey(0);                          //wait infinitly for used imput   (a wait for user input is necessary, otherwise the window will crash)
   
   namedWindow("Gy", WINDOW_AUTOSIZE);
   imshow("Gy", gy);
   waitKey(0);
   
   //devide in blocks.
   if(imWidth % blockSize != 0 || imHeight % blockSize != 0)     //check whether the blocks fit in the image
      cout << "HOGDescriptor::getHOG() Warning! The instructed blocksize does not define a perfect grid in the image!\n";
   
   
   for (int blockPX = 0 ;(blockPX + blockSize) < imWidth; blockPX += blockStride){       // move block around
      //cout << "blockx = " << blockPX << "\n";
      for(int blockPY = 0;(blockPY + blockSize) < imHeight; blockPY += blockStride){
         //cout << "blocky = " << blockPY << "\n";
         //init histogram for the block
         vector<float> hist(nBins, 0);
         
         
         //apply cell operations    
         
         for(int cellX = 0; cellX + cellSize < blockSize; cellX += cellStride){  //move cell around
            //cout << "cellx = " << cellX << "\n";
            for(int cellY=0; cellY + cellSize < blockSize; cellY += cellStride){
               //cout << "celly = " << cellY << "\n";
               for(int x = 0; x < cellSize; x++){    //evaluate gradients in cell
                  for(int y = 0; y < cellSize; y++){
                     //cout << "cell pixel " << x << "," << y << "\n";
                     //  cout << "accessing " << blockPX+cellX+x << "," << blockPY+cellY+y << "\n";
                     //cout << "dx\n";
                     uchar dx = gx.at<uchar>(blockPY + cellY + y, blockPX + cellX + x);
                     //cout << "dy\n";
                     uchar dy = gy.at<uchar>(blockPY + cellY + y, blockPX + cellX + x);
                     //cout << "calc abs gradient\n";
                     double absGrad = sqrt(dx * dx + dy * dy);
                     //cout << "calc orientation\n";
                    
                     double theta = atan( (double)dx / (dy == 0 ? 0.00001f : dy));   //a bit fishy. Need fix for dy==0
                     //cout << "theta: " << theta << "\n";
                     if(theta > M_PI)
                        theta -= M_PI;
                     if(theta < 0)
                        theta += M_PI;
                     //add weighted vote
                     int bin = theta / binSize;
                     //cout << "bin: " << bin <<"\n";
                     votes[bin] += absGrad;
                     //add weighted votes to hist  
                     //cout << "next one..\n";
                     
                  }
               }
               //process votes;
               int maxVote = 0;
               for(int v=1; v < nBins; v++){
                  if(votes[v] > votes[maxVote])
                     maxVote = v;
               }
               //cout << maxVote << "\n";
               //cout << "maxVote=" << maxVote << "\n";;
               hist[maxVote]++;
               
            }
         }
         //cout << "push!";
         histograms.push_back(hist);
      }
      
   }
   
   for (uint i=0;i<histograms.size();i++){   //L2-normalization
      double abs=0;
      for(int j = 0; j < nBins;j++){
          abs += histograms[i][j] * histograms[i][j];
      }
      abs = sqrt(abs);
      for(int j = 0; j < nBins; j++){
          if(abs == 0){
            cout << "hogbovw::HOGDescriptor::getHOG() WARNING!: devision by zero at normalization. Making 0.00001f from it.\n";    //fishy.. Any ideas?
            abs = 0.00001;
          }
          histograms[i][j] /= abs;
      }
   }
   
  
   delete votes;
   cout << "DONE\n";
   return histograms;
}

