
#include <csvm/csvm_lbp_descriptor.h>
//DEPRECATED

using namespace std;
using namespace csvm;

LBPDescriptor::LBPDescriptor() {

    //cout << "we reached 1!";
    //here initialize the feature indexes for when we use rotation invariant LBP
    //cout << "LBPdescriptor initialize:" << endl;

    vector<unsigned int> toUniformValues(256, 0);    //initially points from byte value to smallest representation of this byte
    vector<unsigned int> allUniformValues(256, 0);   //tracks which values are used as smallest representation. 


    //cout << "flag1" << endl;

    
    for (int i = 0; i <= 255; ++i) {
        bitset<8> val(i);
        int shift = 0;
        unsigned int smallestVal = val.to_ulong();
        for (unsigned int bIterate = 1; bIterate < 8; ++bIterate) {
            //cout << "at int: " << i << ", " << val << " , smallest:" << smallestVal << " biterate:" << bIterate << endl;
            if (val[0] == 1) {
                val >>= 1;
                val[7] = 1;
            }
            else {
                val >>= 1;
            }
            if (val.to_ulong() < smallestVal) {
                smallestVal = val.to_ulong();
                shift = bIterate;
            }
        }
        toUniformValues[i] = smallestVal;
        allUniformValues[smallestVal] = 1;
    }
    //cout << "flag 2" << endl;
    int accessIndex = 1;

//    vector<int> uniformFeatureIndex(256, 0);
    
    uniformFeatureIndex = vector<int>(256, 0);
    for (int i = 1; i <= 255; ++i) {
        if (allUniformValues[i] == 1) {
            uniformFeatureIndex[i] = accessIndex;
            ++accessIndex;
        }
        else {
            uniformFeatureIndex[i] = uniformFeatureIndex[toUniformValues[i]];
        }
    }

    // /*
    //for(int i=0;i<=255;++i){
    //cout << i << " corresponds to unif value: " << toUniformValues[i] << " , accesses: " << uniformFeatureIndex[i] << endl;
    //}
    /* */

    /*

    uniformFeatureIndex = vector<int>(255, 0);
    uniformFeatureShifts = vector<int>(255, 0);

    for (int i = 1; i <= 255; ++i) {
        bitset<8> val(i);
        //cout << "int i: " << i << " (" << val << ") --> ";

        int shift = 0;
        int smallestVal = val.to_ulong();
        for (unsigned int bIterate = 1; bIterate < 8; ++bIterate) {
            //cout << "at int: " << i << ", " << val << " , smallest:" << smallestVal << " biterate:" << bIterate << endl;
            if (val[0] == 1) {
                val >>= 1;
                val[7] = 1;
            }
            else {
                val >>= 1;
            }
            if (val.to_ulong() < smallestVal) {
                smallestVal = val.to_ulong();
                shift = bIterate;
            }
        }
        uniformFeatureIndex[i] = smallestVal;
        uniformFeatureShifts[i] = shift;
        //val = bitset<8>(smallestVal);
        //cout << val.to_ulong() << " (" << val << ") shift is " << shift << endl;
    }
    */
    //cout << "we reached 2!";
}

void LBPDescriptor::setSettings(LBPSettings s) {
    //cout << "settings lbp settings" << endl;
    settings = s;

    if (s.uniform == LUNIFORM) {
        settings.LBPSize = 36;
        /*
        vector<int> toUniformValues(255, 0);    // tounif[i]=b , where i is entered lbp value, and b is corresponding uniform value
        vector<int> allUniformValues(255, 0);   //tracks which values are used as smallest representation. 

        for (int i = 0; i <= 255; ++i) {
            bitset<8> val(i);
            int shift = 0;
            unsigned int smallestVal = val.to_ulong();
            for (unsigned int bIterate = 1; bIterate < 8; ++bIterate) {
                //cout << "at int: " << i << ", " << val << " , smallest:" << smallestVal << " biterate:" << bIterate << endl;
                if (val[0] == 1) {
                    val >>= 1;
                    val[7] = 1;
                }
                else {
                    val >>= 1;
                }
                if (val.to_ulong() < smallestVal) {
                    smallestVal = val.to_ulong();
                    shift = bIterate;
                }
            }
            toUniformValues[i] = smallestVal;
            allUniformValues[smallestVal] = 1;
        }
        int accessIndex = 1;
        vector<int> uniformFeatureIndex(255, 0);
        for (int i = 1; i <= 255; ++i) {
            if (allUniformValues[i] == 1) {
                uniformFeatureIndex[i] = accessIndex;
                ++accessIndex;
            }
            else {
                uniformFeatureIndex[i] = uniformFeatureIndex[toUniformValues[i]];
            }
        }

        /*
        for(int i=0;i<=255;++i){
        cout << i << " corresponds to unif value: " << toUniformValues[i] << " , accesses: " << accessIndexes[i] << endl;
        }
        /* */






        /*
        settings.LBPSize = 36;
        vector<int> toUniformValues(255, 0);    //initially points from byte value to smallest representation of this byte
        vector<int> allUniformValues(255, 0);   //tracks which values are used as smallest representation. 

        for (int i = 0; i <= 255; ++i) {
            bitset<8> val(i);
            int shift = 0;
            unsigned int smallestVal = val.to_ulong();
            for (unsigned int bIterate = 1; bIterate < 8; ++bIterate) {
                //cout << "at int: " << i << ", " << val << " , smallest:" << smallestVal << " biterate:" << bIterate << endl;
                if (val[0] == 1) {
                    val >>= 1;
                    val[7] = 1;
                }
                else {
                    val >>= 1;
                }
                if (val.to_ulong() < smallestVal) {
                    smallestVal = val.to_ulong();
                    shift = bIterate;
                }
            }
            toUniformValues[i] = smallestVal;
            allUniformValues[smallestVal] = 1;
            //cout << val.to_ulong() << " (" << val << ") shift is " << shift << endl;    
        }
        unsigned int sum = 0;
        for (int i = 0; i <= 255; ++i) {
            sum += allUniformValues[i];     //sum accumulates the total of unique uniform LBP there are
        }

        vector<int> orderedListArrayIndexes(255, 0);
        unsigned int accessElement = 0;
        unsigned int largestAccessor = 0;
        for (int i = 0; i <= 255; ++i) {
            if (allUniformValues[i] == 1) {   //if this is a uniform value, which is larger than those we have encountered thus far...
                orderedListArrayIndexes[i] = accessElement;
                ++accessElement;
            }
        }
        //cout << "total of " << sum << " uniform values" << endl;
        for (int i = 0; i <= 255; ++i) {
            uniformFeatureIndex[i] = orderedListArrayIndexes[toUniformValues[i]];
            cout << "origin int: " << i << " --> " << toUniformValues[i] << " --> " << orderedListArrayIndexes[toUniformValues[i]] << endl;
        }
        /* */
    }
    else {
        settings.LBPSize = 256;
        uniformFeatureIndex = vector<int>(256, 0);
        for (int i = 0; i <= 255; ++i) {
            uniformFeatureIndex[i] = i;
        }
    }
    
    //cout << "LBPsettings: \n cellSize:" << s.cellSize << "\n cellstride: " << s.cellStride << "\n patchsize: " << s.patchSize << "\n padding: " << s.padding << "\n uniformity: " << s.uniform << "\n usecolour: " << s.useColourPixel << "\npadding: " << s.padding << endl;
    /* */
}


//this function is used to check whether a certain value is uniform or not. 
bool LBPDescriptor::isUniform(int lbp) {
    //cout << "isUniform called";
    int numberOfTrans = 0;
    bitset<8> lbpbits(lbp);
    for (int idx = 0; idx < 7; ++idx) {
        if (lbpbits[idx] != lbpbits[idx + 1]){
            ++numberOfTrans;
        }
    }
    return (numberOfTrans <= 2);
}



//is used to check the uniform equivalent of a function. 
int LBPDescriptor::uniformValue(int lbp) {
    //cout << "uniformvalue called with " << lbp << '\n';
    //iterate down a byte
    int uniformval = 0;
    int lbpvalue = lbp;
    for (int bit = 7; bit > 0; --bit) {
        //if we are at a 1-bit
        if (lbpvalue - pow(2, bit) >= 0) {
            lbpvalue -= pow(2, bit);
            uniformval += (bit*(bit-1)) + 2;
        }
    }
    //cout << "uniformvalue returned " << uniformval << '\n';
    return uniformval;
}

/*
unsigned int LBPDescriptor::computeLBP(unsigned int x, unsigned int y, Patch& patch) {
    
    bitset<(8)> pixelFeatures;

    int centroidPixelIntensity = patch.getGreyPixel(x, y);

    //in a neighbourhood around the centroid pixel: 
    pixelFeatures[0] = ((centroidPixelIntensity > patch.getGreyPixel(x - 1, y - 1)) ? 0 : 1);
    pixelFeatures[1] = ((centroidPixelIntensity > patch.getGreyPixel(x, y - 1)) ? 0 : 1);
    pixelFeatures[2] = ((centroidPixelIntensity > patch.getGreyPixel(x + 1, y - 1)) ? 0 : 1);
    pixelFeatures[3] = ((centroidPixelIntensity > patch.getGreyPixel(x + 1, y)) ? 0 : 1);
    pixelFeatures[4] = ((centroidPixelIntensity > patch.getGreyPixel(x + 1, y + 1)) ? 0 : 1);
    pixelFeatures[5] = ((centroidPixelIntensity > patch.getGreyPixel(x, y + 1)) ? 0 : 1);
    pixelFeatures[6] = ((centroidPixelIntensity > patch.getGreyPixel(x - 1, y + 1)) ? 0 : 1);
    pixelFeatures[7] = ((centroidPixelIntensity > patch.getGreyPixel(x - 1, y)) ? 0 : 1);
    return pixelFeatures.to_ulong;
}
/**/

int LBPDescriptor::lbpdiff(unsigned int centX, unsigned int centY, unsigned int X, unsigned int Y, Patch patch, LBPColour col) {
    if (col == LGRAY) {
        int centroidPixelIntensity = patch.getGreyPixel(centX, centY);
        //cout << " (" << X << "," << Y << ") " << ((centroidPixelIntensity > patch.getGreyPixel(X, Y)) ? 0 : 1) << " from intensity: " << patch.getGreyPixel(X, Y) << endl;
        return ((centroidPixelIntensity > patch.getGreyPixel(X, Y)) ? 0 : 1);
    }
    else {
        int centroidPixelIntensity = patch.getPixel(centX, centY, col);
        //cout << " (" << X << "," << Y << ") " << ((centroidPixelIntensity > patch.getPixel(X, Y, col)) ? 0 : 1) << " from intensity: " << patch.getPixel(X, Y, col) << endl;
        return ((centroidPixelIntensity > patch.getPixel(X, Y, col)) ? 0 : 1);
    }

}

unsigned int LBPDescriptor::computeLBP(unsigned int x, unsigned int y, Patch patch, LBPColour col) {
    
    bitset<(8)> pixelFeatures;

        pixelFeatures[0] = lbpdiff(x,y, x - 1, y - 1, patch, col);
        pixelFeatures[1] = lbpdiff(x, y, x, y - 1, patch, col);
        pixelFeatures[2] = lbpdiff(x, y, x+1, y - 1, patch, col);
        pixelFeatures[3] = lbpdiff(x, y, x+1, y, patch, col);
        pixelFeatures[4] = lbpdiff(x, y, x+1, y + 1, patch, col);
        pixelFeatures[5] = lbpdiff(x, y, x, y + 1, patch, col);
        pixelFeatures[6] = lbpdiff(x, y, x-1, y + 1, patch, col);
        pixelFeatures[7] = lbpdiff(x, y, x-1, y, patch, col);


        //cout << "computing flaglagl lbp of pixel (" << (x) << "," << (y) << "), has center intensity" << patch.getGreyPixel(x, y) << endl;


    //cout << "lbp is: " << pixelFeatures.to_ulong() << endl;
    //cout << "let's print uniformfeatureindex" << uniformFeatureIndex[0] << "and 255: " << uniformFeatureIndex[255] << endl;
    //cout << " , binned into bin: " << uniformFeatureIndex[ ((unsigned int)pixelFeatures.to_ulong())] << endl;

    return (unsigned int)pixelFeatures.to_ulong();
}

void LBPDescriptor::binLBP(unsigned int X, unsigned int Y, LBPColour col, vector<double>& cellLBPHistogram, Patch block) {
    //here we deal with 
    //cout << "computing lbp of (" << X << "," << Y << ")\n";
    //cout << "computing lbp of pixel (" << (X) << "," << (Y) << "), has center intensity" << ( settings.useColourPixel ? block.getPixel(X,Y,col):block.getGreyPixel(X, Y)) << ", with lbp= ";
    unsigned int lbpval = computeLBP(X, Y, block, col);
    //cout << lbpval << " , binned into bin: " << uniformFeatureIndex[lbpval] << endl;
    
    ++cellLBPHistogram[uniformFeatureIndex[ lbpval ]];
}

vector<double> LBPDescriptor::computeCellLBP(unsigned int cellX, unsigned int cellY, Patch patch) {
    //cout << "computing cell lbp of X:" << cellX << ", Y: " << cellY << endl;

    vector<double> returnLBP(0, 0.0);
    
    if (settings.binmethod == LCROSSCOLOUR) {
        //cout << "crosscolour binning" << endl;
        vector <double> cellOrientationHistogram(settings.LBPSize, 0.0);

            for (size_t X = 0; X < settings.cellSize; ++X)
            {
                for (size_t Y = 0; Y < settings.cellSize; ++Y)
                {
                    if (!settings.useColourPixel) {
                        //cout << "no colours" << endl;
                        binLBP(X + cellX, Y + cellY, LGRAY, cellOrientationHistogram, patch);

                    }
                    else {
                        binLBP(X + cellX, Y + cellY, LRED, cellOrientationHistogram, patch);
                        //binPixel(X + cellX, Y + cellY, GREEN, cellOrientationHistogram, block);
                        binLBP(X + cellX, Y + cellY, LGREEN, cellOrientationHistogram, patch);
                        //binPixel(X + cellX, Y + cellY, BLUE, cellOrientationHistogram, block);
                        binLBP(X + cellX, Y + cellY, LBLUE, cellOrientationHistogram, patch);
                    }
                }
            }



            returnLBP = cellOrientationHistogram;
    }
    else {

        vector <double> cellOrientationHistogram(settings.LBPSize, 0); //if we bin cross-colour, then we'll only have single HOG, if we bin by-colour, then we'll have multiple HOGS appended to one another.
        vector <double> redCellOrientationHistogram(settings.LBPSize, 0); //if we bin cross-colour, then we'll only have single HOG, if we bin by-colour, then we'll have multiple HOGS appended to one another.
        vector <double> blueCellOrientationHistogram(settings.LBPSize, 0); //if we bin cross-colour, then we'll only have single HOG, if we bin by-colour, then we'll have multiple HOGS appended to one another.
        vector <double> greenCellOrientationHistogram(settings.LBPSize, 0); //if we bin cross-colour, then we'll only have single HOG, if we bin by-colour, then we'll have multiple HOGS appended to one another.

        for (size_t X = 0; X < settings.cellSize; ++X) {
            for (size_t Y = 0; Y < settings.cellSize; ++Y) {
                binLBP(X + cellX, Y + cellY, LRED, redCellOrientationHistogram, patch);
                binLBP(X + cellX, Y + cellY, LGREEN, greenCellOrientationHistogram, patch);
                binLBP(X + cellX, Y + cellY, LBLUE, blueCellOrientationHistogram, patch);
            }
        }
        returnLBP.insert(returnLBP.end(), redCellOrientationHistogram.begin(), redCellOrientationHistogram.end());
        returnLBP.insert(returnLBP.end(), greenCellOrientationHistogram.begin(), greenCellOrientationHistogram.end());
        returnLBP.insert(returnLBP.end(), blueCellOrientationHistogram.begin(), blueCellOrientationHistogram.end());
    }

    //cout << "printing the lbp of a single cell:" << endl;

    //for (int idx = 0; idx < settings.LBPSize; ++idx) {
        //cout << returnLBP[idx] << " , ";
    //}
    //cout << endl;
    
    //here we deal with what the size of the feature vector must be, in relation to colour and stuff. 

    return returnLBP;
}

Feature LBPDescriptor::getLBP(Patch& patch) {
    int patchWidth = patch.getWidth();
    int patchHeight = patch.getHeight();
    vector<double> LBPHistogram(0, 0);
    settings.patchSize = patch.getHeight();

    int colours = ((settings.useColourPixel) * 2) + 1;

    if (settings.padding == LNONE) {
        //cout << "no padding in use" << endl;
        for (int cellX = 1; cellX + settings.cellSize <= patchWidth-1; cellX += settings.cellStride) {
            for (int cellY = 1; cellY + settings.cellSize <= patchHeight-1; cellY += settings.cellStride) {
                //cout << "cell: " << cellX << ", " << cellY << '\n';

                vector <double> cellOrientationHistogram = computeCellLBP(cellX, cellY, patch);    //cellOrientationHistogram tracks the HOG for the current cell
                //cout << "\ncellorientationHistogram:\n";
                //for (unsigned int ita = 0; ita < cellOrientationHistogram.size(); ++ita) {
                //    cout << cellOrientationHistogram[ita] << " , ";
                //}
                LBPHistogram.insert(LBPHistogram.end(), cellOrientationHistogram.begin(), cellOrientationHistogram.end());
            }
        }
    }
    else {
        for (int cellX = 0; cellX + settings.cellSize <= patchWidth; cellX += settings.cellStride) {
            for (int cellY = 0; cellY + settings.cellSize <= patchHeight; cellY += settings.cellStride) {
                //cout << "cell: " << cellX << ", " << cellY << '\n';

                vector <double> cellOrientationHistogram = computeCellLBP(cellX, cellY, patch);    //cellOrientationHistogram tracks the HOG for the current cell

                LBPHistogram.insert(LBPHistogram.end(), cellOrientationHistogram.begin(), cellOrientationHistogram.end());
            }
        }

    }

    // perhaps some postprocessing here....




    Feature result(LBPHistogram);
    result.label = patch.getLabel();
    result.labelId = patch.getLabelId();

    return result;
}




