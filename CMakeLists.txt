cmake_minimum_required (VERSION 2.6)
project (HOG-BOVW)

add_executable(
HOG-BOVW


src/main.cc
##name .cc files here

src/hogbovw/hb_image_scanner.cc
src/hogbovw/hb_hog_descriptor.cc
)

find_package(OpenCV 3.0.0 EXACT REQUIRED NO_MODULE)

include_directories(${OpenCV_INCLUDE_DIR} "include/")

set_target_properties(HOG-BOVW PROPERTIES RUNTIME_OUTPUT_DIRECTORY "build/")
## set compiler settings heere : 
add_definitions("-Wall")

target_link_libraries(HOG-BOVW ${OpenCV_LIBS} -lm)
