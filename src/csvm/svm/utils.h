# ifndef UTILS_H
# define UTILS_H

# include <stdio.h>
# include <string.h>
# include <iostream>


using namespace std;

//bool verbose;	// Whether or not to print messages
extern bool verbose;

// Standard output/logger wrapper
void report(int cLev, string fnName, string message);







# endif
