
#include <csvm/csvm.h>
#include <iostream>

using namespace csvm;
using namespace std;

int main(int argc,char**argv){
   ImageScanner is(15,100);  //window size, nPatches
   is.setImage("lenna.png");  //load image into ImageScanner using openCV
   is.showImage();
   
   is.scanImage();
   cout << "Hello World!\n";
   return 0;
}






