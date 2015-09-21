typedef struct word_list_struct word_list;
struct word_list_struct {
  char **word;
  int number_of_words;
};

/* utils.c -> world */
extern word_list line_to_words();

