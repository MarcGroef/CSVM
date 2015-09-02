//copyright Marc Groefsema (c) 2015
#include <hogbovw/hogbovw.h>
#include <iostream>

using namespace hogbovw;
using namespace std;

int main(int argc,char**argv){
   ImageScanner is(15,100);
   is.setImage("lenna.png");
   is.showImage();
   
   is.scanImage();
   cout << "Hello World!\n";
   return 0;
}






