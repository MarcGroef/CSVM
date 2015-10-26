#include <csvm/csvm_hog_descriptor.h>
#include <iomanip> //for setprecision couting

using namespace std;
using namespace csvm;


//HOGDescriptor::HOGDescriptor(int nBins = 9, int cellSize = 3, int blockSize = 9, bool useGreyPixel = 1) {
HOGDescriptor::HOGDescriptor() {
   /*this->settings.nBins=9;
   this->settings.cellSize = -1;
   this->settings.cellStride = -1;
   this->settings.blockSize = -1;
   //this->blockStride = -1;
   this->settings.numberOfCells = -1;
   this->settings.useGreyPixel = true;*/
   
}

HOGDescriptor::HOGDescriptor(int cellSize, int cellStride, int blockSize) {
	this->settings.nBins = 9;				//9 bins was shown to perform optimal in previous experiments
	this->settings.cellSize = cellSize;
	this->settings.cellStride = cellStride;
	this->settings.blockSize = blockSize;
	//this->settings.padding = ZERO;
	//this->blockStride = blockStride;
	this->settings.numberOfCells = pow( ((settings.blockSize - settings.cellSize) / settings.cellSize) + 1, 2);
	this->settings.useGreyPixel = true;
}

void HOGDescriptor::setSettings(HOGSettings s){
   settings = s;
   //cout << "hog settigns set\n";
   this->settings.padding = IDENTITY;
   settings.nBins = 9;
   this->settings.numberOfCells = pow( ((settings.blockSize - settings.cellSize) / settings.cellSize) + 1, 2);
   this->settings.useGreyPixel = false;
}

double HOGDescriptor::computeXGradient(Patch patch, int x, int y, Colour col = GRAY) {
	double result;
	if (settings.useGreyPixel) {
		double xPlus = (x + 1 > patch.getWidth() ? settings.padding*patch.getGreyPixel(x, y) : patch.getGreyPixel(x + 1, y));
		double xMin = (x - 1 < 0 ? settings.padding*patch.getGreyPixel(x, y) : patch.getGreyPixel(x - 1, y));
		result = xPlus - xMin;
	}
	else
	{
		double xPlus = (x + 1 > patch.getWidth() ? settings.padding*patch.getPixel(x, y, col) : patch.getPixel(x + 1, y, col));
		double xMin = (x - 1 < 0 ? settings.padding*patch.getPixel(x, y, col) : patch.getPixel(x - 1, y, col));
		result = xPlus - xMin;
	}//patch.getPixel(x,y, channel in 0-2)
	return result;
}

double HOGDescriptor::computeYGradient(Patch patch, int x, int y, Colour col) {
	double result=0;
	//for now, implement zero padding hardcoded
	if (settings.useGreyPixel) {
		double yPlus = (y + 1 > patch.getHeight() ? settings.padding*patch.getGreyPixel(x, y) : patch.getGreyPixel(x, y+1));
		double yMin = (y - 1 < 0 ? settings.padding*patch.getGreyPixel(x, y) : patch.getGreyPixel(x, y-1));
		result = yPlus - yMin;
	}
	else
	{
		double yPlus = (y + 1 > patch.getHeight() ? settings.padding*patch.getPixel(x, y, col) : patch.getPixel(x, y + 1, col));
		double yMin = (y - 1 < 0 ? settings.padding*patch.getPixel(x, y, col) : patch.getPixel(x, y - 1, col));
		result = yPlus - yMin;
	}
	return result;
}

double HOGDescriptor::computeMagnitude(double xGradient, double yGradient) {
	return sqrt(pow(xGradient, 2) + pow(yGradient, 2));
}

double HOGDescriptor::computeOrientation(double xGradient, double yGradient) {
   double ori = atan2(yGradient, xGradient) * 180.0 / M_PI;
   while(ori < 0.0)
      ori += 180.0;
   
	return ori;
}

//This function implements classic HOG, including how to partitionize the image. CSVM will do this in another way, so it's not quite finished
Feature HOGDescriptor::getHOG(Patch block,int channel, bool useGreyPixel=1){
   
   vector <double> gx,gy;
   unsigned int patchWidth = block.getWidth();
   unsigned int patchHeight = block.getHeight();

   if (patchWidth % 2 == 1 || patchHeight % 2 == 1 || patchHeight != patchWidth) {
	   cout << "patch size is wrong! It is " << patchWidth << " x " << patchHeight << '\n';
   }
 
  
  
   vector <double> blockHistogram(0, 0);
  
   //iterate through block with a cell, with stride cellstride. 
     
   for (int cellX = 0; cellX + settings.cellSize <= patchWidth; cellX += settings.cellStride) {
	   for (int cellY = 0; cellY+ settings.cellSize <= patchHeight; cellY += settings.cellStride) {
		   //cout << "cell: " << cellX << ", " << cellY << '\n';

		   vector <double> cellOrientationHistogram(settings.nBins, 0);
         //cout << "cellOri set\n";
		   //now for every cell, compute histogram of features. 
		   //adjust for padding type. if no padding, then only iterate over an offset of boundary
		   for (size_t X = (settings.padding == NONE ? 1 : 0); X < (settings.padding == NONE ? settings.cellSize - 1 : settings.cellSize); ++X)
		   {
			   for (size_t Y = (settings.padding == NONE ? 1 : 0); Y < (settings.padding == NONE ? settings.cellSize - 1 : settings.cellSize); ++Y)
			   {
				   double xGradient;
				   double yGradient;
				   double gradientMagnitude;
				   double gradientOrientation;
				   size_t bin;

				   if (settings.useGreyPixel)
				   {
					   xGradient = computeXGradient(block, X + cellX, Y + cellY, GRAY);
					   yGradient = computeYGradient(block, X + cellX, Y + cellY, GRAY);
					   gradientMagnitude = computeMagnitude(xGradient, yGradient);
					   gradientOrientation = computeOrientation(xGradient, yGradient);
					   bin = (unsigned int)(gradientOrientation / (180.0 / settings.nBins));
					   cellOrientationHistogram[bin > settings.nBins - 1 ? settings.nBins - 1 : bin] += gradientMagnitude;
				   }
				   else
				   {
					   xGradient = computeXGradient(block, X + cellX, Y + cellY, RED);
					   yGradient = computeYGradient(block, X + cellX, Y + cellY, RED);
					   gradientMagnitude = computeMagnitude(xGradient, yGradient);
					   gradientOrientation = computeOrientation(xGradient, yGradient);
					   bin = (unsigned int)(gradientOrientation / (180.0 / settings.nBins));
					   cellOrientationHistogram[bin > settings.nBins - 1 ? settings.nBins - 1 : bin] += gradientMagnitude;

					   xGradient = computeXGradient(block, X + cellX, Y + cellY, GREEN);
					   yGradient = computeYGradient(block, X + cellX, Y + cellY, GREEN);
					   gradientMagnitude = computeMagnitude(xGradient, yGradient);
					   gradientOrientation = computeOrientation(xGradient, yGradient);
					   bin = (unsigned int)(gradientOrientation / (180.0 / settings.nBins));
					   cellOrientationHistogram[bin > settings.nBins - 1 ? settings.nBins - 1 : bin] += gradientMagnitude;

					   xGradient = computeXGradient(block, X + cellX, Y + cellY, BLUE);
					   yGradient = computeYGradient(block, X + cellX, Y + cellY, BLUE);
					   gradientMagnitude = computeMagnitude(xGradient, yGradient);
					   gradientOrientation = computeOrientation(xGradient, yGradient);
					   bin = (unsigned int)(gradientOrientation / (180.0 / settings.nBins));
					   cellOrientationHistogram[bin > settings.nBins - 1 ? settings.nBins - 1 : bin] += gradientMagnitude;

				   }


               //cout << "added to histogram\n";
			   }
		   }
		   //cout << "\nsingle cell feature vect:" << '\n';
		   //for (size_t idx = 0; idx < settings.nBins; ++idx) {
			//   cout << std::setprecision(3) << cellOrientationHistogram[idx] << " | " ;
		   //}
		   //cout << "made it to histogram insertion\n";
		   //now we have fully processed a single cell. let's append it to our to-be feature vector.
		   blockHistogram.insert(blockHistogram.end(), cellOrientationHistogram.begin(), cellOrientationHistogram.end());
		   //cout << "\nblockHistogram: " << '\n';
		   //for (size_t idx = 0; idx < blockHistogram.size(); ++idx) {
		//	   cout << blockHistogram[idx] << " | ";
		  // }
	   }
   }
   //now we have processed all cells, whose histograms are all appended to one another in blockHistrogram. 
   //now we should normalize it 
   
   //we first implement normalization
   //first find the lowest and highest value for a bin:
   /*
   double lowestValue = 360;
   double highestValue = 0;
   for (size_t idx = 0; idx < blockHistogram.size(); ++idx) {
		highestValue = (blockHistogram[idx] > highestValue ? blockHistogram[idx] : highestValue);
		lowestValue = (blockHistogram[idx] < highestValue ? blockHistogram[idx] : lowestValue);
   }
   */
	//L2 normalization scheme:
   double vTwoSquared = 0;
   for (size_t idx = 0; idx < blockHistogram.size(); ++idx) {
	   vTwoSquared += pow(blockHistogram[idx], 2);
   }
	//   vTwoSquared = sqrt(vTwoSquared); //is now vector length

   // e is some magic number still...
   double e = 0.01;
   for (size_t idx = 0; idx < blockHistogram.size(); ++idx) {
	   blockHistogram[idx] /= sqrt(vTwoSquared + pow(e, 2));
   }

   //Feature result(settings.nBins*settings.numberOfCells, 0);
   Feature result(blockHistogram.size(), 0);
   result.content = blockHistogram;
   result.label = block.getLabel();
   result.labelId = block.getLabelId();
   //cout << "HOG passed the label " << result.labelId << endl;
   //cout << "returning feature with size " << result.content.size() << endl;
   return result;
   /*
   vector< vector<double> > histograms;   //collection of HOG histograms
   double* votes = new double[nBins];
   double binSize = M_PI/nBins;
   


		   //in a neighbourhood around the centroid pixel: 
		   histogram.content[(int)pixelFeatures.to_ulong()] += 1;
	   }
   }
   return histogram;
   */


   //get gradients of image
   /*
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
   */
}

