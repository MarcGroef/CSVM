cmake_minimum_required (VERSION 2.6)
project (CSVM)

add_executable(
CSVM


#add all .c/.cc files here, otherwise they wont be compiled and linked

src/main.cc
##name .cc files here

src/csvm/csvm_image.cc
src/csvm/csvm_image_scanner.cc
src/csvm/csvm_hog_descriptor.cc
src/csvm/csvm_frequency_matrix.cc
src/csvm/csvm_cifar10_parser.cc
src/csvm/csvm_dataset.cc
src/csvm/csvm_classifier.cc
src/csvm/csvm_settings.cc
src/csvm/csvm_patch.cc
src/csvm/csvm_rbm.cc
src/csvm/csvm_lbp_descriptor.cc
src/csvm/csvm_feature.cc
src/csvm/csvm_codebook.cc
src/csvm/csvm_lvq.cc
src/csvm/csvm_kmeans.cc
src/csvm/csvm_cluster_analyser.cc
src/csvm/csvm_feature_extractor.cc
src/csvm/csvm_clean_descriptor.cc
src/csvm/csvm_svm.cc

##png library
src/lodepng/lodepng.cc

##dnn project files
src/dnn/dnn_data.c
src/dnn/dnn_flow.c
src/dnn/dnn_layer_stack.c
src/dnn/dnn_math.c
src/dnn/dnn_pretrainer.c
src/dnn/dnn_weights.c

)


#define a place where #include<..> can search from
include_directories(
   "include/"
   )

set_target_properties(CSVM PROPERTIES RUNTIME_OUTPUT_DIRECTORY "build/")
## set compiler settings here : 

#-Wall instructs g++ the show all compile warnings. There should be no compile warnings.
#-g causes the linker to link c++ code with the machinecode, such that e.g. valgrind can give directed messeges towards your code, which is nice :)

add_definitions("-Wall -g -O3")   

#link the std math library to the program
target_link_libraries( CSVM
   -lm
)
