cmake_minimum_required (VERSION 2.6)
project (CSVM)

add_executable(
CSVM




#add all .c/.cc files here, otherwise they wont be compiled and linked

src/main.cc
##name .cc files here
src/csvm/csvm_experiment.cc
src/csvm/csvm_image.cc
src/csvm/csvm_image_scanner.cc
src/csvm/csvm_hog_descriptor.cc
src/csvm/csvm_frequency_matrix.cc
src/csvm/csvm_cifar10_parser.cc
src/csvm/csvm_dataset.cc
src/csvm/csvm_classifier.cc
src/csvm/csvm_settings.cc
src/csvm/csvm_patch.cc
#src/csvm/csvm_rbm.cc
src/csvm/csvm_lbp_descriptor.cc
src/csvm/csvm_feature.cc
src/csvm/csvm_codebook.cc
src/csvm/csvm_lvq.cc
src/csvm/csvm_kmeans.cc
#src/csvm/csvm_cluster_analyser.cc
src/csvm/csvm_feature_extractor.cc
src/csvm/csvm_clean_descriptor.cc
src/csvm/csvm_svm.cc
src/csvm/csvm_mnist_parser.cc
src/csvm/csvm_linear_network.cc
src/csvm/csvm_centroid.cc
src/csvm/csvm_deep_codebook.cc
src/csvm/csvm_annotated_kmeans.cc
src/csvm/csvm_conv_svm.cc
src/csvm/csvm_merge_descriptor.cc

##png library
src/lodepng/lodepng.cc

##dnn project files
##src/dnn/dnn_data.c
#src/dnn/dnn_flow.c
##src/dnn/dnn_layer_stack.c
#src/dnn/dnn_math.c
#src/dnn/dnn_pretrainer.c
#src/dnn/dnn_weights.c

)

#find_package(SWIG REQUIRED)
#include(${SWIG_USE_FILE})

#find_package(PythonLibs)
#include_directories(${PYTHON_INCLUDE_PATH})
#include_directories("include/")
#set(CMAKE_SWIG_FLAGS "")
#set_source_files_properties(csvm.i PROPERTIES CPLUSPLUS ON)
#set_source_files_properties(csvm.i PROPERTIES SWIG_FLAGS "-includeall")

#SWIG_ADD_MODULE(csvm python csvm.i csvm_wrap.cxx)
#SWIG_LINK_LIBRARIES(csvm ${PYTHON_LIBRARIES})

#define a place where #include<..> can search from
include_directories(
   "include/"
   )

set_target_properties(CSVM PROPERTIES RUNTIME_OUTPUT_DIRECTORY "build/")
## set compiler settings here : 

#-Wall instructs g++ the show all compile warnings. There should be no compile warnings.
#-g causes the linker to link c++ code with the machinecode, such that e.g. valgrind can give directed messeges towards your code, which is nice :)

add_definitions("-Wall -g -O2 -c")   

#link the std math library to the program
target_link_libraries( CSVM
   -lm
)
