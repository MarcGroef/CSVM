/* input:  line
   output: array of words, function returns amount of words
   ALLOCATES MEMORY NEEDS malloc.h
   The amount of words in the line should not exceed MAX_WORDS
*/

#include <cstdlib>
#include <math.h>
#include <limits.h>
#include <stdio.h>
#include "utils.h"


#define MAX_WORDS 1000

using namespace std;

/* extract all words from a line */
word_list line_to_words(char *line) {
  
  int i, j;
  int read;
  int word_length[MAX_WORDS];
  word_list words;

  /* initialization */
  for (i=0; i<MAX_WORDS; i++)
    word_length[i] = 0;
  i=0;
  words.number_of_words = 0;

  /* first pass: calculate the amount of words and the word length
     needed to allocate memory */
  while (line[i] != '\n') {
    read = 0;
    while (((line[i] == ' ') || (line[i] == '\t') || (line[i] == ','))  
	   && (line[i] != '\n')) /* ignore blanks and tabs */
	      i++;
    while ((line[i] != ' ') && (line[i] != '\t') && (line[i] != ',')
	   && (line[i] != '\n')) { /* calculate word length */
      word_length[words.number_of_words]++;
      i++;
      read = 1; /* needed in order to ignore blanks at the end of a line */
    }
    if (read == 1) {
      words.number_of_words++;
    }
  }

  /* memory allocation */
  if (NULL == (words.word 
	       = (char **) calloc(words.number_of_words, sizeof(char*)))) {
    printf("line_to_words: Could not allocate memory...\n");
    //exit(1);
  }
  for (i=0; i<words.number_of_words; i++)
    if (NULL == (words.word[i] 
		 = (char *) calloc(word_length[i]+1, sizeof(char)))) {
      printf("line_to_words: Could not allocate memory...\n");
      //exit(1);
    }      

  /* reinitialization */
  i=0;
  words.number_of_words = 0;

  /* second pass: extract the words */
  while (line[i] != '\n') {
    read = 0;
    while (((line[i] == ' ') || (line[i] == '\t') || (line[i] == ',')) 
	   && (line[i] != '\n')) /* ignore blanks and tabs */
	      i++;
    j = 0;
    while ((line[i] != ' ') && (line[i] != '\t') && (line[i] != ',') 
	   && (line[i] != '\n')) { /* extract word */
      words.word[words.number_of_words][j] = line[i];
      i++;
      j++;
      read = 1; /* needed in order to ignore blanks at the end of a line */
    }
    if (read == 1) {
      words.word[words.number_of_words][j] = '\0';
      words.number_of_words++;
    }
  }

  return(words);
  
} /* line_to_words */

