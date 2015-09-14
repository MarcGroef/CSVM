#include <csvm/csvm_hog_descriptor.h>


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


//This function implements classic HOG, including how to partitionize the image. CSVM will do this in another way, so it's not quite finished
vector< vector<double> > HOGDescriptor::getHOG(Image image,int channel){
   Image gx,gy;
   gx = image.clone();  //cloned image to store horizontal gradient
   gy = image.clone();  //cloned image to store vertical gradient
   
   int imWidth = image.getWidth();
   int imHeight = image.getHeight();
   
   vector< vector<double> > histograms;   //collection of HOG histograms
   double* votes = new double[nBins];
   double binSize = M_PI/nBins;
   
   if(imWidth != imHeight)
      cout << "HOGDescriptor::getHOG() Warning! Image is not a square! Some parts might not get scanned.\n";
   
   //get gradients of image
   for(int i=0; i<imWidth; i++)
      for(int j=0;j < imHeight; j++){
         int gv=(i -1 >= 0 ? image.getPixel(j, i - 1,channel) : 0) - (i + 1 < imWidth ? image.getPixel(j, i + 1,channel): 0);    //calculate difference between left and right pixel
         gx.setPixel(j,i,channel,(unsigned char)abs(gv));                                                    //store absolute difference
         gv = (j - 1 >= 0 ? image.getPixel(j - 1, i,channel) : 0)-(j+1<imHeight?image.getPixel(j + 1, i,channel) : 0);       //same for up and down
         gy.setPixel(j,i,channel,(unsigned char)abs(gv));
      }
      
      
   //show the gradients for fun
   for(int i=0;i<imWidth;i++)
      for(int j=0; j<imHeight;j++)
         for(int c=0;c<3;c++){
            if(c==channel) continue;
            gx.setPixel(i,j,c,0);
            gy.setPixel(i,j,c,0);
         }
      
   
   gx.exportImage("gx.png");
   gy.exportImage("gy.png");
   
   
   
   
   //devide in blocks.
   if(imWidth % blockSize != 0 || imHeight % blockSize != 0)     //check whether the blocks fit in the image
      cout << "HOGDescriptor::getHOG() Warning! The instructed blocksize does not define a perfect grid in the image!\n";
   
   
   for (int blockPX = 0 ;(blockPX + blockSize) < imWidth; blockPX += blockStride){       // move block around
      //cout << "blockx = " << blockPX << "\n";
      for(int blockPY = 0;(blockPY + blockSize) < imHeight; blockPY += blockStride){
         //cout << "blocky = " << blockPY << "\n";
         //init histogram for the block
         vector<double> hist(nBins, 0);
         
         
         //apply cell operations    
         
         for(int cellX = 0; cellX + cellSize < blockSize; cellX += cellStride){  //move cell around
            //cout << "cellx = " << cellX << "\n";
            for(int cellY=0; cellY + cellSize < blockSize; cellY += cellStride){
               //cout << "celly = " << cellY << "\n";
               //reset votes
               for(int v=0;v<nBins;v++)
                  votes[v]=0;
               for(int x = 0; x < cellSize; x++){    //evaluate gradients in cell
                  for(int y = 0; y < cellSize; y++){
                     //cout << "cell pixel " << x << "," << y << "\n";
                     //  cout << "accessing " << blockPX+cellX+x << "," << blockPY+cellY+y << "\n";
                     //cout << "dx\n";
                     unsigned char dx = gx.getPixel(blockPY + cellY + y, blockPX + cellX + x,channel);
                     //cout << "dy\n";
                     unsigned char dy = gy.getPixel(blockPY + cellY + y, blockPX + cellX + x,channel);
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
                     int bin =(int)( theta / binSize);
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
   
   for (unsigned int i=0;i<histograms.size();i++){   //L2-normalization
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
   
  
   delete[] votes;
   cout << "DONE\n";
   return histograms;
}

