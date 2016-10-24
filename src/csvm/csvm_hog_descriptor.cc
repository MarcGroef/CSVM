#include <csvm/csvm_hog_descriptor.h>
#include <iomanip> //for setprecision couting
#include <limits>
using namespace std;
using namespace csvm;

HOGDescriptor::HOGDescriptor() {
   /*
HOGSettings default;
FeatureExtractor
method HOG
nBins 9
cellSize 6
cellStride 6
patchSize 12
padding Identity
useColourPixel true
interpolation INTERPOLATE_LINEAR
binmethod CROSSCOLOUR
postprocessing l2norm
debugLevel 0
   */
}

void HOGDescriptor::setSettings(HOGSettings s) {
   settings = s;
      //so here, we're making sure that our input is okay, and that HOG can do well with parameter optimization algorithms

   if (s.cellSize > s.patchSize) {  //cellsize shouldn't be larger than patchsize
      if (s.debugLevel >= 0)  
         cout << "FYI, cellsizes larger than patchSizes is not really the best of ideas..., you might want to revise that. but for the sake of the run, I'll set the cellSize to the patchSize, with stride 0" << endl;
      s.cellSize = s.patchSize;
      s.cellStride = 1; // This should probably be done in order for the for loop to be able to deal with this stuff.
   }
   if (s.cellSize < s.cellStride) { //it's unlogical for cellsize to be larger than it's own stride..
      if (s.debugLevel >= 0) 
         cout << "cellsize > cellstride...? you'll be skipping pixels here. I hope you don't mind if I adjust it to cellStride = cellSize..";
      s.cellStride = s.cellSize;
   }
   if (s.cellSize <= 0 || s.patchSize <= 2 || s.cellStride <= 0) {   //patchsize should be bigger than just 1 x 1, cellSize should be bigger than 0, stride too
      if (s.debugLevel >= 0) 
         cout << "your HOG patch and cell sizes and strides were (seriously) out of acceptable ranges: pSize" << s.patchSize << ", cellsize: " << s.cellSize << " stride: " << s.cellStride << ", check them again\n";
      if (s.debugLevel != -1) {
         exit(-1);
      }
      else { //if it is REALLY essential that no output is shown, or any output would break the system, then we override the set values, and pray to glob that they can be used.
         s.cellSize = 2;
         s.cellStride = 1;
      }
   }

   //should we give an errr or some feedback whenever any property is not set? Or when it is inconsistent values?
   if (s.padding != NONE) { //with some type of padding that uses the entire patch...
      //here we should ensure that the cells, in combination with the cellstride, fit exactly within the patch size
      if (((s.patchSize - s.cellSize) % s.cellStride) != 0) {
         if (s.debugLevel >= 0) 
            cout << "patchsize, or cellsize/stride is inconsistent! overriding cellSize and stride for fit\n";
         //here we'd want to make an adjustment, probably to the cell size or stride, as the patch size is quite more hard-set.
         s.cellStride = s.cellSize;
         while (((s.patchSize - s.cellSize) % s.cellStride) != 0) {
            --s.cellStride;
         }
         //now we have a cellSize that still conforms to the size the user indicated, but with different cellstride as such that it'll fit, adjusted as such that it'll fit exactly within the patch
         
                     //s.cellSize = (s.patchSize % 2 == 0 ? s.patchSize / 2 : s.patchSize / 3); //adjust cellSize to make it fit exactly in the patch
                     //s.cellStride = s.cellSize;  //adjust cellStride accordingly. No overlap.
                     //if (((s.patchSize - s.cellSize) % s.cellStride) != 0) {   //if sizes are still inconsistent, we might be dealing with some sort of prime numbered patchSize
                     // s.cellSize = s.patchSize - 1;
                     // s.cellStride = 1; //in that case, we FIX IT WITH ONE SIMPLE TRICK! CLICK HERE TO FIND OUT MORE AND CLAIM YOUR PRIZE THAT MAKES DOCTORS FURIOUS AT HER!
                     //}

         if (s.debugLevel >=0) 
            cout <<  "overridden cellStride: " << s.cellStride << endl;
      }
   }
   else {
      //padding is NONE, so we work with a effectively useable patch size which is 2 less from all borders, so:
      
      if (s.patchSize <= 7 || (s.cellSize + s.cellStride > s.patchSize - 2)) {
         if (s.debugLevel >=0) 
            cout << "you're using NONE padding with unlogical patch size, or cellsizes, readjusting for fit" << endl;
         s.cellStride = s.cellSize;
         while ((((s.patchSize-2) - s.cellSize) % s.cellStride) != 0) {
            --s.cellStride;
         }
         if (s.debugLevel >=0) 
            cout << "adjusted values: cellSize:" << s.cellSize << ", cellStride" << s.cellStride << endl;
      }
      
      if ((((s.patchSize - 2) - s.cellSize) % s.cellStride) != 0){
         if (s.debugLevel >=0) 
            cout << "patchsize, or cellsize/stride is inconsistent! You're using no padding, so keep in mind that your useable patch size is size-2 \n";
         if (s.debugLevel >=0) 
            cout << "adjusting to fit" << endl;

         s.cellStride = s.cellSize;
         while (((s.patchSize - s.cellSize) % s.cellStride) != 0) {
            --s.cellStride;
         }
         if (s.debugLevel >=0) 
            cout << "overridden cellStride: " << s.cellStride << endl;
         //let's hope this works...
               //if ((((s.patchSize-2) - s.cellSize) % s.cellStride) != 0) {  //if sizes are still inconsistent, we might be dealing with some sort of prime numbered patchSize
               // s.cellSize = (s.patchSize-2) - 1;
               // s.cellStride = 1; //in that case, we FIX IT WITH ONE SIMPLE TRICK! CLICK HERE TO FIND OUT MORE AND CLAIM YOUR PRIZE THAT MAKES DOCTORS FURIOUS AT HER!
               //}

      }
   }

   //cout << "hog settigns set\n";
}

float HOGDescriptor::computeXGradient(Patch patch, int x, int y, Colour col) {
   float result;
   if (!settings.useColourPixel || col == GRAY) {
      float xPlus = (x + 1 == patch.getWidth() ? settings.padding*patch.getGreyPixel(x, y) : patch.getGreyPixel(x + 1, y));
      float xMin = (x - 1 < 0 ? settings.padding*patch.getGreyPixel(x, y) : patch.getGreyPixel(x - 1, y));
      result = xPlus - xMin;
   }
   else
   {
      float xPlus = (x + 1 == patch.getWidth() ? settings.padding*patch.getPixel(x, y, col) : patch.getPixel(x + 1, y, col));
      float xMin = (x - 1 < 0 ? settings.padding*patch.getPixel(x, y, col) : patch.getPixel(x - 1, y, col));
      result = xPlus - xMin;
   }//patch.getPixel(x,y, channel in 0-2)
   return result;
}

float HOGDescriptor::computeYGradient(Patch patch, int x, int y, Colour col) {
   float result;
   if (!settings.useColourPixel || col == GRAY) {
      float yPlus = (y + 1 == patch.getHeight() ? settings.padding*patch.getGreyPixel(x, y) : patch.getGreyPixel(x, y + 1));
      float yMin = (y - 1 < 0 ? settings.padding*patch.getGreyPixel(x, y) : patch.getGreyPixel(x, y - 1));
      result = yPlus - yMin;
   }
   else
   {
      float yPlus = (y + 1 == patch.getHeight() ? settings.padding*patch.getPixel(x, y, col) : patch.getPixel(x, y + 1, col));
      float yMin = (y - 1 < 0 ? settings.padding*patch.getPixel(x, y, col) : patch.getPixel(x, y - 1, col));
      result = yPlus - yMin;
   }
   return result;
}

float HOGDescriptor::computeMagnitude(float xGradient, float yGradient) {
   return sqrt(pow(xGradient, 2) + pow(yGradient, 2));
}

float HOGDescriptor::computeOrientation(float xGradient, float yGradient) {
   float ori = atan2(yGradient, xGradient) * 180.0 / M_PI;
   while (ori < 0.0)
      ori += 180.0;
   return ori;
}

void HOGDescriptor::binPixel(size_t X, size_t Y, Colour col, vector<float>& cellOrientationHistogram, Patch& block) {
   //cout << "binPixel called for x:" << X << ", y:" << Y << "\n";

   float xGradient = computeXGradient(block, X, Y, col);
   float yGradient = computeYGradient(block, X, Y, col);
   float gradientMagnitude = computeMagnitude(xGradient, yGradient);
   float gradientOrientation = computeOrientation(xGradient, yGradient);
   if (settings.interpol == INTERPOLATE_BINARY) {
      size_t bin = (unsigned int)(gradientOrientation / (180.0 / settings.nBins));
      cellOrientationHistogram[bin == settings.nBins ? 0 : bin] += gradientMagnitude;
   }
   else if (settings.interpol == INTERPOLATE_LINEAR) {
      int bandWidth = (int)(180.0 / settings.nBins); // = 20
      int orientationBin = (int)gradientOrientation; // = 58
      int base = bandWidth / 2; // = 10 , first bin value, also added mid values of bins


      int lowerBinIndex = ((orientationBin - base + 180) % 180) / 20;
      int upperBinIndex = lowerBinIndex == (int)(settings.nBins - 1) ? 0 : (lowerBinIndex + 1);

      //cout << "lwrbID " << lowerBinIndex << " | " << upperBinIndex << " hgherbID" << endl;

      int lowerBinValue = lowerBinIndex*bandWidth + base; // = 58 - 18 + 10 = 40 + 10 = bin 50 (aka: 40-60)
      //int upperBinValue = upperBinIndex*bandWidth + base; // = 50+20 = bin 70 (aka: 60-80)
                                             //cout << "before: lowerbin: " << lowerBinIndex << ", upperbin: " << upperBinIndex << "\n";
      float lowerBinAddedValue = gradientMagnitude*(1.0 - (((gradientOrientation - lowerBinValue) > 0 ? (gradientOrientation - lowerBinValue) : (180 + (orientationBin - lowerBinValue))) / bandWidth));

      float higherBinAddedValue = gradientMagnitude*(((gradientOrientation - lowerBinValue) > 0 ? (gradientOrientation - lowerBinValue) : (180 + (orientationBin - lowerBinValue))) / bandWidth);

      //cout << "before: lowerbin: " << lowerBinIndex << ", upperbin: " << upperBinIndex << "\n";

      cellOrientationHistogram[lowerBinIndex] += lowerBinAddedValue;
      cellOrientationHistogram[upperBinIndex] += higherBinAddedValue;
   }
}

void HOGDescriptor::binPixel(size_t X, size_t Y, Colour col, vector<float>& cellOrientationHistogram, float ****imageTranspose) {
   float gradientOrientation;
   float gradientMagnitude;
   

   //cout << "fancy binPixel called for x:" << X << ", y:" << Y << "\n";
   if (col == GRAY || !settings.useColourPixel) {
      //cout << "we are using grey colour, somehow. either entered col was grey, or settings is not colourpixel" << endl;
      gradientOrientation = imageTranspose[0][X][Y][ORIENTATION];
      //cout << "gradient looked up was" << gradientOrientation << '\n';
      gradientMagnitude = imageTranspose[0][X][Y][MAGNITUDE];
   }
   else {
      //cout << "we are using colours" << endl;
      gradientOrientation = imageTranspose[col][X][Y][ORIENTATION];
      //cout << "gradient looked up was" << gradientOrientation << '\n';
      gradientMagnitude = imageTranspose[col][X][Y][MAGNITUDE];
   }
   if (settings.interpol == INTERPOLATE_BINARY) {
      size_t bin = (unsigned int)(gradientOrientation / (180.0 / settings.nBins));
      cellOrientationHistogram[bin == settings.nBins ? 0 : bin] += gradientMagnitude;
   }
   else if (settings.interpol == INTERPOLATE_LINEAR) {      //with 9 | 18 bins:
      float bandWidth = (180.0 / settings.nBins); // = 180 / 9 | 18 = 20 | 10
      float orientationBin = gradientOrientation; // = 58
      float base = bandWidth / 2; // = 10 | 5 , first bin value, also added mid values of bins


      int lowerBinIndex = (int)((orientationBin - base < 0 ? (orientationBin - base) + 180.0 : orientationBin - base) / bandWidth );      //int lowerBinIndex = ((orientationBin - base + 180) % 180) / bandWidth;
      int upperBinIndex = lowerBinIndex == (int)(settings.nBins - 1) ? 0 : (lowerBinIndex + 1);

      //cout << "lwrbID " << lowerBinIndex << " | " << upperBinIndex << " hgherbID" << endl;

      float lowerBinValue = lowerBinIndex*bandWidth + base; // = 58 - 18 + 10 = 40 + 10 = bin 50 (aka: 40-60)
      float upperBinValue = upperBinIndex*bandWidth + base; // = 50+20 = bin 70 (aka: 60-80)
      // << "before: lowerbin: " << lowerBinIndex << ", upperbin: " << upperBinIndex << "\n";
      if (gradientOrientation == lowerBinValue) {
         cellOrientationHistogram[lowerBinIndex] += gradientMagnitude;
      }
      else {
         float lowerBinAddedValue = gradientMagnitude*(1.0 - (((gradientOrientation - lowerBinValue) > 0 ? (gradientOrientation - lowerBinValue) : (180 + (orientationBin - lowerBinValue))) / bandWidth));
         if (settings.debugLevel >=2) cout << col << ": added a share of " << lowerBinAddedValue << " to lowerbin " << lowerBinIndex << " (" << lowerBinValue << " )";

         float higherBinAddedValue = gradientMagnitude*(((gradientOrientation - lowerBinValue) > 0 ? (gradientOrientation - lowerBinValue) : (180 + (orientationBin - lowerBinValue))) / bandWidth);
         if (settings.debugLevel >=2) cout << "and a share of " << higherBinAddedValue << " to higherbin " << upperBinIndex << " (" << upperBinValue << " )\n";
         //cout << "before: lowerbin: " << lowerBinIndex << ", upperbin: " << upperBinIndex << "\n";

         cellOrientationHistogram[lowerBinIndex] += lowerBinAddedValue;
         cellOrientationHistogram[upperBinIndex] += higherBinAddedValue;
      }
   }
}

float ****HOGDescriptor::patchTranspose(Patch& block, float ****transposedImage, unsigned int colours) {

   vector <float> gx, gy;
   unsigned int patchWidth = block.getWidth();
   unsigned int patchHeight = block.getHeight();

   //here we must create the transposed of the patch, which contains all orientations and magnitudes for every pixel, for every colour channel. 
   //we must take into consideration how padding is done here, especially as it would affect 
   if (settings.padding != NONE) {
      if (settings.useColourPixel == true) {
         //cout << "making transposed image\n";
         transposedImage = new float***[colours]; //= total size to store compelte image array-wise
         float xGradient = 0.0;
         float yGradient = 0.0;

         transposedImage[RED] = new float**[patchWidth];
         transposedImage[GREEN] = new float**[patchWidth];
         transposedImage[BLUE] = new float**[patchWidth];

         for (unsigned int X = 0; X < patchWidth; ++X) {
            transposedImage[RED][X] = new float*[patchHeight];
            transposedImage[GREEN][X] = new float*[patchHeight];
            transposedImage[BLUE][X] = new float*[patchHeight];


            for (unsigned int Y = 0; Y < patchHeight; ++Y) {
               //Red          
               transposedImage[RED][X][Y] = new float[2];
               xGradient = computeXGradient(block, X, Y, RED);
               yGradient = computeYGradient(block, X, Y, RED);
               transposedImage[RED][X][Y][MAGNITUDE] = computeMagnitude(xGradient, yGradient);
               transposedImage[RED][X][Y][ORIENTATION] = computeOrientation(xGradient, yGradient);
               //green
               transposedImage[GREEN][X][Y] = new float[2];
               xGradient = computeXGradient(block, X, Y, GREEN);
               yGradient = computeYGradient(block, X, Y, GREEN);
               transposedImage[GREEN][X][Y][MAGNITUDE] = computeMagnitude(xGradient, yGradient);
               transposedImage[GREEN][X][Y][ORIENTATION] = computeOrientation(xGradient, yGradient);
               //blue
               transposedImage[BLUE][X][Y] = new float[2];
               xGradient = computeXGradient(block, X, Y, BLUE);
               yGradient = computeYGradient(block, X, Y, BLUE);
               transposedImage[BLUE][X][Y][MAGNITUDE] = computeMagnitude(xGradient, yGradient);
               transposedImage[BLUE][X][Y][ORIENTATION] = computeOrientation(xGradient, yGradient);

            }
         }

         //cout << "done making transposed \n";

      }
      else {   //if we're not working with colour. 
         transposedImage = new float***[1]; //= total size to store compelte image array-wise
         float xGradient = 0.0;
         float yGradient = 0.0;
         transposedImage[0] = new float**[patchWidth];
         for (unsigned int X = 0; X < patchWidth; ++X) {
            transposedImage[0][X] = new float*[patchHeight];

            for (unsigned int Y = 0; Y < patchHeight; ++Y) {
               //gray
               transposedImage[0][X][Y] = new float[2];
               xGradient = computeXGradient(block, X, Y, GRAY);
               yGradient = computeYGradient(block, X, Y, GRAY);
               transposedImage[0][X][Y][MAGNITUDE] = computeMagnitude(xGradient, yGradient);
               transposedImage[0][X][Y][ORIENTATION] = computeOrientation(xGradient, yGradient);
            }
         }
      }
   }
   else {         // we're working with padding = NONE
      if (settings.useColourPixel == true) {
         //cout << "making transposed image\n";
         transposedImage = new float***[colours]; //= total size to store compelte image array-wise
         float xGradient = 0.0;
         float yGradient = 0.0;

         transposedImage[RED] = new float**[patchWidth - 2]; //our arrays are smaller
         transposedImage[GREEN] = new float**[patchWidth - 2];
         transposedImage[BLUE] = new float**[patchWidth - 2];

         for (unsigned int X = 1; X < patchWidth - 1; ++X) {   //we start and end within the boundary
            transposedImage[RED][X - 1] = new float*[patchHeight - 2]; //but we must access our arrays accordingly, 
            transposedImage[GREEN][X - 1] = new float*[patchHeight - 2];
            transposedImage[BLUE][X - 1] = new float*[patchHeight - 2];


            for (unsigned int Y = 1; Y < patchHeight - 1; ++Y) {
               //Red          
               transposedImage[RED][X - 1][Y - 1] = new float[2];
               xGradient = computeXGradient(block, X, Y, RED); //here we give the exact coordinates, so we give them through correctly
               yGradient = computeYGradient(block, X, Y, RED);
               transposedImage[RED][X - 1][Y - 1][MAGNITUDE] = computeMagnitude(xGradient, yGradient);
               transposedImage[RED][X - 1][Y - 1][ORIENTATION] = computeOrientation(xGradient, yGradient);
               //green
               transposedImage[GREEN][X - 1][Y - 1] = new float[2];
               xGradient = computeXGradient(block, X, Y, GREEN);
               yGradient = computeYGradient(block, X, Y, GREEN);
               transposedImage[GREEN][X - 1][Y - 1][MAGNITUDE] = computeMagnitude(xGradient, yGradient);
               transposedImage[GREEN][X - 1][Y - 1][ORIENTATION] = computeOrientation(xGradient, yGradient);
               //blue
               transposedImage[BLUE][X - 1][Y - 1] = new float[2];
               xGradient = computeXGradient(block, X, Y, BLUE);
               yGradient = computeYGradient(block, X, Y, BLUE);
               transposedImage[BLUE][X - 1][Y - 1][MAGNITUDE] = computeMagnitude(xGradient, yGradient);
               transposedImage[BLUE][X - 1][Y - 1][ORIENTATION] = computeOrientation(xGradient, yGradient);

            }
         }

         //cout << "done making transposed \n";

      }
      else {   //if we're not working with colour. 
         transposedImage = new float***[1]; //= total size to store compelte image array-wise
         float xGradient = 0.0;
         float yGradient = 0.0;
         transposedImage[0] = new float**[patchWidth - 2];
         for (unsigned int X = 1; X < patchWidth - 1; ++X) {
            transposedImage[0][X - 1] = new float*[patchHeight - 2];

            for (unsigned int Y = 1; Y < patchHeight - 1; ++Y) {
               //gray
               transposedImage[0][X - 1][Y - 1] = new float[2];
               xGradient = computeXGradient(block, X, Y, GRAY);
               yGradient = computeYGradient(block, X, Y, GRAY);
               transposedImage[0][X - 1][Y - 1][MAGNITUDE] = computeMagnitude(xGradient, yGradient);
               transposedImage[0][X - 1][Y - 1][ORIENTATION] = computeOrientation(xGradient, yGradient);
            }
         }
      }
   }
   return transposedImage;
}



vector <float> HOGDescriptor::computeCellHOG(float ****imageTranspose, unsigned int cellX, unsigned int cellY) {
   //let's compute the HOG for a single cell.

   vector <float> returnHOG(0, 0.0);//(settings.nBins) * ((settings.binmethod * 2) + 1), 0.0); //if we bin cross-colour, then we'll only have single HOG, if we bin by-colour, then we'll have multiple HOGS appended to one another.


   if (settings.binmethod == CROSSCOLOUR) { //we bin the colours into a single histogram
   
      vector <float> cellOrientationHistogram(settings.nBins, 0); //if we bin cross-colour, then we'll only have single HOG, if we bin by-colour, then we'll have multiple HOGS appended to one another.


      for (size_t X = 0; X < settings.cellSize; ++X)
      {
         for (size_t Y = 0; Y < settings.cellSize; ++Y)
         {
            if (settings.debugLevel >= 2) {
               cout << "\ncellOrient before crosscolour binning: \n";
               //grey pixels only have to be binned into a single channel. 
               for (unsigned int ita = 0; ita < cellOrientationHistogram.size(); ++ita) {
                  cout << cellOrientationHistogram[ita] << " , ";
               }
               cout << "\nbinning data: \n";
            }
            //cout << "are we using grey pixels?" << endl;
            if (!settings.useColourPixel)
            {
               //cout << "yes we are using grey pixels" << endl;
               
               if (settings.debugLevel >=2)
                  cout << "\npixel (" << X + cellX << "," << Y + cellY << ") , M:" << imageTranspose[0][X + cellX][Y + cellY][MAGNITUDE] << " , G:" << imageTranspose[0][X + cellX][Y + cellY][ORIENTATION] << "\n";

               binPixel(X + cellX, Y + cellY, GRAY, cellOrientationHistogram, imageTranspose);

            }
            else
            {
               if (settings.debugLevel >= 2) {
                  cout << "\npixel (" << X + cellX << "," << Y + cellY << ") , of Red(MAG:" << imageTranspose[RED][X + cellX][Y + cellY][MAGNITUDE] << ", GRAD:" << imageTranspose[RED][X + cellX][Y + cellY][ORIENTATION] << "\n";
                  cout << "pixel (" << X + cellX << "," << Y + cellY << ") , of Green(MAG:" << imageTranspose[GREEN][X + cellX][Y + cellY][MAGNITUDE] << ", GRAD:" << imageTranspose[GREEN][X + cellX][Y + cellY][ORIENTATION] << "\n";
                  cout << "pixel (" << X + cellX << "," << Y + cellY << ") , of Blue(MAG:" << imageTranspose[BLUE][X + cellX][Y + cellY][MAGNITUDE] << ", GRAD:" << imageTranspose[BLUE][X + cellX][Y + cellY][ORIENTATION] << "\n";
               }
               //binPixel(X + cellX, Y + cellY, RED, cellOrientationHistogram, block);
               binPixel(X + cellX, Y + cellY, RED, cellOrientationHistogram, imageTranspose);
               //binPixel(X + cellX, Y + cellY, GREEN, cellOrientationHistogram, block);
               binPixel(X + cellX, Y + cellY, GREEN, cellOrientationHistogram, imageTranspose);
               //binPixel(X + cellX, Y + cellY, BLUE, cellOrientationHistogram, block);
               binPixel(X + cellX, Y + cellY, BLUE, cellOrientationHistogram, imageTranspose);
            }
            if (settings.debugLevel >= 2) {
               cout << "\ncellOrient after binning: \n";

               for (unsigned int ita = 0; ita < cellOrientationHistogram.size(); ++ita) {
                  cout << cellOrientationHistogram[ita] << " , ";
               }
               cout << endl;
            }
            //exit(-1);
         }
      }
      returnHOG = cellOrientationHistogram;
   }
   else {      //we are binning by colours, so for every colour, we create a seperate HOG, which are appended to one another in the end. 

      vector <float> cellOrientationHistogram(settings.nBins, 0); //if we bin cross-colour, then we'll only have single HOG, if we bin by-colour, then we'll have multiple HOGS appended to one another.
      vector <float> redCellOrientationHistogram(settings.nBins, 0); //if we bin cross-colour, then we'll only have single HOG, if we bin by-colour, then we'll have multiple HOGS appended to one another.
      vector <float> blueCellOrientationHistogram(settings.nBins, 0); //if we bin cross-colour, then we'll only have single HOG, if we bin by-colour, then we'll have multiple HOGS appended to one another.
      vector <float> greenCellOrientationHistogram(settings.nBins, 0); //if we bin cross-colour, then we'll only have single HOG, if we bin by-colour, then we'll have multiple HOGS appended to one another.

      if (!settings.useColourPixel)
      {

         for (size_t X = 0; X < settings.cellSize; ++X)
         {
            for (size_t Y = 0; Y < settings.cellSize; ++Y)
            {
               //actually, we'd never come here. why bin by colour if you're not using colours?..
               //cout << "are we using grey pixels?" << endl;

                  //cout << "yes we are using grey pixels" << endl;
               binPixel(X + cellX, Y + cellY, GRAY, cellOrientationHistogram, imageTranspose);
            }
         }
         returnHOG = cellOrientationHistogram;
      }
      else {
         for (size_t X = 0; X < settings.cellSize; ++X) {
            for (size_t Y = 0; Y < settings.cellSize; ++Y) {
               if (settings.debugLevel >= 2) {
                  cout << "\ncellOrient before BYcolour binning: ";
                  //grey pixels only have to be binned into a single channel. 
                  cout << "\nred:  ";
                  for (unsigned int itr = 0; itr < redCellOrientationHistogram.size(); ++itr) {
                     cout << redCellOrientationHistogram[itr] << " , ";
                  }
                  cout << "\ngreen:";
                  for (unsigned int itg = 0; itg < greenCellOrientationHistogram.size(); ++itg) {
                     cout << greenCellOrientationHistogram[itg] << " , ";
                  }
                  cout << "\nblue: ";
                  for (unsigned int itb = 0; itb < blueCellOrientationHistogram.size(); ++itb) {
                     cout << blueCellOrientationHistogram[itb] << " , ";
                  }
                  cout << endl;
                  cout << "\npixel (" << X + cellX << "," << Y + cellY << ") , of Red(MAG:" << imageTranspose[RED][X + cellX][Y + cellY][MAGNITUDE] << ", GRAD:" << imageTranspose[RED][X + cellX][Y + cellY][ORIENTATION] << "\n";
                  cout << "pixel (" << X + cellX << "," << Y + cellY << ") , of Green(MAG:" << imageTranspose[GREEN][X + cellX][Y + cellY][MAGNITUDE] << ", GRAD:" << imageTranspose[GREEN][X + cellX][Y + cellY][ORIENTATION] << "\n";
                  cout << "pixel (" << X + cellX << "," << Y + cellY << ") , of Blue(MAG:" << imageTranspose[BLUE][X + cellX][Y + cellY][MAGNITUDE] << ", GRAD:" << imageTranspose[BLUE][X + cellX][Y + cellY][ORIENTATION] << "\n";
                  //binPixel(X + cellX, Y + cellY, RED, cellOrientationHistogram, block);
                  cout << "red:  ";
               }
               binPixel(X + cellX, Y + cellY, RED, redCellOrientationHistogram, imageTranspose);
               //binPixel(X + cellX, Y + cellY, GREEN, cellOrientationHistogram, block);
               if (settings.debugLevel >=2) cout << "green:";
               binPixel(X + cellX, Y + cellY, GREEN, greenCellOrientationHistogram, imageTranspose);
               //binPixel(X + cellX, Y + cellY, BLUE, cellOrientationHistogram, block);
               if (settings.debugLevel >=2) cout << "blue: ";
               binPixel(X + cellX, Y + cellY, BLUE, blueCellOrientationHistogram, imageTranspose);

               if (settings.debugLevel >= 2) {
                  cout << "\ncellOrient after byColour binning: \n";

                  //grey pixels only have to be binned into a single channel. 
                  cout << "\nred:  ";
                  for (unsigned int itr = 0; itr < redCellOrientationHistogram.size(); ++itr) {
                     cout << redCellOrientationHistogram[itr] << " , ";
                  }
                  cout << "\ngreen:";
                  for (unsigned int itg = 0; itg < greenCellOrientationHistogram.size(); ++itg) {
                     cout << greenCellOrientationHistogram[itg] << " , ";
                  }
                  cout << "\nblue: ";
                  for (unsigned int itb = 0; itb < blueCellOrientationHistogram.size(); ++itb) {
                     cout << blueCellOrientationHistogram[itb] << " , ";
                  }
                  //exit(-1);
               }
            }
         }
         returnHOG.insert(returnHOG.end(), redCellOrientationHistogram.begin(), redCellOrientationHistogram.end());
         returnHOG.insert(returnHOG.end(), greenCellOrientationHistogram.begin(), greenCellOrientationHistogram.end());
         returnHOG.insert(returnHOG.end(), blueCellOrientationHistogram.begin(), blueCellOrientationHistogram.end());
      }
   }

   //nBins * ((patchSize - CellSize) / CellStride)+1)^2 * 3*useColourPixel*byColourBinning

   return returnHOG;
}


vector <float> HOGDescriptor::postProcess(vector <float> blockHistogram) {


   //POSTPROCESsING ------------------------------------------

   //now we have processed all cells, whose histograms are all appended to one another in blockHistrogram. 
   //now we should normalize it 

   //we first implement normalization
   //first find the lowest and highest value for a bin:
   if (settings.postproccess == PURE) {
      if (settings.debugLevel >= 1) cout << "performing pure postprocessing. no adjustments made." << endl;
      return blockHistogram;
   }
   if (settings.postproccess == STANDARDISATION) {
      float mean = 0.0;
      float standardDeviation = 0.0;
      size_t featureLen = blockHistogram.size() ;

      for (size_t idx = 0; idx < featureLen; ++idx) {
         mean += blockHistogram[idx];
      }
      mean /= (float)featureLen;
      float deviation = 0.0;
      for (size_t idx = 0; idx < featureLen; ++idx) {
         deviation += pow((blockHistogram[idx] - mean), 2.0);
      }
      deviation /= (float)featureLen;
      standardDeviation = sqrt(deviation);

      for (size_t idx = 0; idx < featureLen; ++idx) {
         blockHistogram[idx] = (blockHistogram[idx] - mean) / standardDeviation;
      }
      if (settings.debugLevel >= 1) {
         cout << "performing standardisation, mean:" << mean << " , std: " << standardDeviation << endl;
      }
      return blockHistogram;
   }
   if (settings.postproccess == NORMALISATION) {

      size_t featureLen = blockHistogram.size();
      float max = 0.0;
      float min = numeric_limits<float>::max();
      for (size_t idx = 0; idx < featureLen; ++idx) {
         max = blockHistogram[idx] > max ? blockHistogram[idx] : max;
         min = blockHistogram[idx] < min ? blockHistogram[idx] : min;
      }
      //updating content with normalized values:
      float diff = max - min;
      for (size_t idx = 0; idx < featureLen; ++idx) {
         blockHistogram[idx] = (blockHistogram[idx] - min) / (diff);
      }
      if (settings.debugLevel >= 1) {
         cout << "performing normalisation with max:" << max << " , min:" << min << endl;
      }
      return blockHistogram;
   }

   if (settings.postproccess == LTWONORM) {
      //L2 normalization scheme:
      float vTwoSquared = 0;
      for (size_t idx = 0; idx < blockHistogram.size(); ++idx) {
         vTwoSquared += pow(blockHistogram[idx], 2);
      }
      //   vTwoSquared = sqrt(vTwoSquared); //is now vector length

      // e is some magic number still...
      float e = 0.000001;
      for (size_t idx = 0; idx < blockHistogram.size(); ++idx) {
         blockHistogram[idx] /= sqrt(vTwoSquared + pow(e, 2));
      }
      if (settings.debugLevel >= 1) {
         cout << "performing lTwoNorm, v2Squared:" << vTwoSquared << endl;
      }
      return blockHistogram;
   }

   if (settings.postproccess == CLIPNORM) {
      float lowestValue = 360;
      float highestValue = 0;
      for (size_t idx = 0; idx < blockHistogram.size(); ++idx) {
         highestValue = (blockHistogram[idx] > highestValue ? blockHistogram[idx] : highestValue);
         lowestValue = (blockHistogram[idx] < highestValue ? blockHistogram[idx] : lowestValue);
      }

      //L2 normalization scheme:
      float vTwoSquared = 0;
      for (size_t idx = 0; idx < blockHistogram.size(); ++idx) {
         vTwoSquared += pow(blockHistogram[idx], 2);
      }
      //   vTwoSquared = sqrt(vTwoSquared); //is now vector length

      // e is some magic number still...
      float e = 0.000001;
      for (size_t idx = 0; idx < blockHistogram.size(); ++idx) {
         blockHistogram[idx] /= sqrt(vTwoSquared + pow(e, 2));
      }


      //0.2 clipping as in paper http://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf
      for (size_t idx = 0; idx < blockHistogram.size(); ++idx) {
         blockHistogram[idx] = blockHistogram[idx] > 0.2 ? 0.2 : blockHistogram[idx];
      }

      //and normalize again
      vTwoSquared = 0;
      for (size_t idx = 0; idx < blockHistogram.size(); ++idx) {
         vTwoSquared += pow(blockHistogram[idx], 2);
      }
      //   vTwoSquared = sqrt(vTwoSquared); //is now vector length

      // e is some magic number still...
      //cout << "Feature :\n";
      for (size_t idx = 0; idx < blockHistogram.size(); ++idx) {
         blockHistogram[idx] /= sqrt(vTwoSquared + pow(e, 2));
         //cout << blockHistogram[idx] << endl;
      }

      return blockHistogram;
   }

   if (settings.debugLevel >= 0) cout << "somehow the postprocessing type wasn't indicated, no regularisation performed. \n";
   return blockHistogram; //default case
}



//This function implements classic HOG, and is the function called when computing the HOG-feature of an incoming patch (which we call block)
Feature HOGDescriptor::getHOG(Patch& block) {
   vector <float> gx, gy;
   unsigned int patchWidth = block.getWidth();
   unsigned int patchHeight = block.getHeight();
   settings.patchSize = block.getHeight();
   vector <float> blockHistogram(0, 0);

   int colours = ((settings.useColourPixel) * 2) + 1; //how many colours do we have? if useColour == true, then 3, otherwise just 1
   //cout << "we have " << colours << " colours\n";
   float**** transposedImage; //transposedImage [ colour channel ] [X-coordinate] [Y-coordinate] [MAGNITUDE or ORIENTATION]
   transposedImage = patchTranspose(block, transposedImage, colours);   //compute the orientations and magnitudes of all pixels, and save these to a new map.


   //now we have to iterate over this field by a cell, and for every cell bin all magnitudes in a HOG:
   if (settings.padding == NONE) { //adjust patchSizes when working with the transposed while using No padding
      patchWidth = patchWidth - 2;
      patchHeight = patchHeight - 2;
   }

   for (int cellX = 0; cellX + settings.cellSize <= patchWidth; cellX += settings.cellStride) {
      for (int cellY = 0; cellY + settings.cellSize <= patchHeight; cellY += settings.cellStride) {
         //cout << "cell: " << cellX << ", " << cellY << '\n';

         vector <float> cellOrientationHistogram = computeCellHOG(transposedImage, cellX, cellY);    //cellOrientationHistogram tracks the HOG for the current cell
         
         if (settings.debugLevel >= 1) {
            cout << "\ncellorientationHistogram:\n";
            for (unsigned int ita = 0; ita < cellOrientationHistogram.size(); ++ita) {
               cout << cellOrientationHistogram[ita] << " , ";
            }

         }
         //now we have fully processed a single cell. let's append it to our to-be feature vector.
         blockHistogram.insert(blockHistogram.end(), cellOrientationHistogram.begin(), cellOrientationHistogram.end());
      }
   }

   if (settings.debugLevel >= 1) {
      cout << "\nfeature vector before postprocessing:\n";
      for (unsigned int ita = 0; ita < blockHistogram.size(); ++ita) {
         cout << blockHistogram[ita] << " , ";
         if (ita != 0 && ((ita+1) % settings.nBins == 0)) {
            cout << endl;
         }
      }
      cout << "\nperforming postProcessing: \n";
   }
   //POSTPROCESsING ------------------------------------------
   
   //now we have processed all cells, whose histograms are all appended to one another in blockHistrogram. 
   blockHistogram = postProcess(blockHistogram);

   if (settings.debugLevel >= 1) {
      cout << "\nfeature vector AFTER postprocessing:\n";
      for (unsigned int ita = 0; ita < blockHistogram.size(); ++ita) {
         cout << blockHistogram[ita] << " , ";
         if (ita != 0 && ((ita+1) % settings.nBins == 0)) {
            cout << endl;
         }
      }
      cout << endl;
   }
   //Feature result(settings.nBins*settings.numberOfCells, 0);
   Feature result(blockHistogram);
   result.label = block.getLabel();
   result.labelId = block.getLabelId();

   ///cleanup transposed image
   for (int co = 0; co < colours; ++co) {
      for (unsigned int X = 0; X < patchWidth; ++X) {
         for (unsigned int Y = 0; Y < patchHeight; ++Y) {
            delete[] transposedImage[co][X][Y];    //delete magnitude and orientation
         }
         delete[] transposedImage[co][X];          //delete full y array
      }
      delete[] transposedImage[co];          //delete full x array
   }
   delete[] transposedImage;        //delete col array

   return result;
}

