cmake_minimum_required (VERSION 2.6)
project (HOG-BOVW)

add_executable(
CSVM


src/main.cc
##name .cc files here

src/csvm/csvm_image_scanner.cc
src/csvm/csvm_hog_descriptor.cc
src/csvm/csvm_frequency_matrix.cc
)

find_package(OpenCV 3.0.0 EXACT REQUIRED NO_MODULE)

include_directories(${OpenCV_INCLUDE_DIR} "include/")

set_target_properties(CSVM PROPERTIES RUNTIME_OUTPUT_DIRECTORY "build/")
## set compiler settings here : 
add_definitions("-Wall")

target_link_libraries(CSVM ${OpenCV_LIBS} -lm)
