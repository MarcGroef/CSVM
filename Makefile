# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

# Default target executed when no arguments are given to make.
default_target: all
.PHONY : default_target

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/cmake-gui

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/marc/bachelor/CSVM

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/marc/bachelor/CSVM

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/bin/cmake-gui -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache
.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache
.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/marc/bachelor/CSVM/CMakeFiles /home/marc/bachelor/CSVM/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/marc/bachelor/CSVM/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean
.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named CSVM

# Build rule for target.
CSVM: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 CSVM
.PHONY : CSVM

# fast build rule for target.
CSVM/fast:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/build
.PHONY : CSVM/fast

src/csvm/csvm_cifar10_parser.o: src/csvm/csvm_cifar10_parser.cc.o
.PHONY : src/csvm/csvm_cifar10_parser.o

# target to build an object file
src/csvm/csvm_cifar10_parser.cc.o:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_cifar10_parser.cc.o
.PHONY : src/csvm/csvm_cifar10_parser.cc.o

src/csvm/csvm_cifar10_parser.i: src/csvm/csvm_cifar10_parser.cc.i
.PHONY : src/csvm/csvm_cifar10_parser.i

# target to preprocess a source file
src/csvm/csvm_cifar10_parser.cc.i:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_cifar10_parser.cc.i
.PHONY : src/csvm/csvm_cifar10_parser.cc.i

src/csvm/csvm_cifar10_parser.s: src/csvm/csvm_cifar10_parser.cc.s
.PHONY : src/csvm/csvm_cifar10_parser.s

# target to generate assembly for a file
src/csvm/csvm_cifar10_parser.cc.s:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_cifar10_parser.cc.s
.PHONY : src/csvm/csvm_cifar10_parser.cc.s

src/csvm/csvm_classifier.o: src/csvm/csvm_classifier.cc.o
.PHONY : src/csvm/csvm_classifier.o

# target to build an object file
src/csvm/csvm_classifier.cc.o:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_classifier.cc.o
.PHONY : src/csvm/csvm_classifier.cc.o

src/csvm/csvm_classifier.i: src/csvm/csvm_classifier.cc.i
.PHONY : src/csvm/csvm_classifier.i

# target to preprocess a source file
src/csvm/csvm_classifier.cc.i:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_classifier.cc.i
.PHONY : src/csvm/csvm_classifier.cc.i

src/csvm/csvm_classifier.s: src/csvm/csvm_classifier.cc.s
.PHONY : src/csvm/csvm_classifier.s

# target to generate assembly for a file
src/csvm/csvm_classifier.cc.s:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_classifier.cc.s
.PHONY : src/csvm/csvm_classifier.cc.s

src/csvm/csvm_clean_descriptor.o: src/csvm/csvm_clean_descriptor.cc.o
.PHONY : src/csvm/csvm_clean_descriptor.o

# target to build an object file
src/csvm/csvm_clean_descriptor.cc.o:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_clean_descriptor.cc.o
.PHONY : src/csvm/csvm_clean_descriptor.cc.o

src/csvm/csvm_clean_descriptor.i: src/csvm/csvm_clean_descriptor.cc.i
.PHONY : src/csvm/csvm_clean_descriptor.i

# target to preprocess a source file
src/csvm/csvm_clean_descriptor.cc.i:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_clean_descriptor.cc.i
.PHONY : src/csvm/csvm_clean_descriptor.cc.i

src/csvm/csvm_clean_descriptor.s: src/csvm/csvm_clean_descriptor.cc.s
.PHONY : src/csvm/csvm_clean_descriptor.s

# target to generate assembly for a file
src/csvm/csvm_clean_descriptor.cc.s:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_clean_descriptor.cc.s
.PHONY : src/csvm/csvm_clean_descriptor.cc.s

src/csvm/csvm_cluster_analyser.o: src/csvm/csvm_cluster_analyser.cc.o
.PHONY : src/csvm/csvm_cluster_analyser.o

# target to build an object file
src/csvm/csvm_cluster_analyser.cc.o:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_cluster_analyser.cc.o
.PHONY : src/csvm/csvm_cluster_analyser.cc.o

src/csvm/csvm_cluster_analyser.i: src/csvm/csvm_cluster_analyser.cc.i
.PHONY : src/csvm/csvm_cluster_analyser.i

# target to preprocess a source file
src/csvm/csvm_cluster_analyser.cc.i:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_cluster_analyser.cc.i
.PHONY : src/csvm/csvm_cluster_analyser.cc.i

src/csvm/csvm_cluster_analyser.s: src/csvm/csvm_cluster_analyser.cc.s
.PHONY : src/csvm/csvm_cluster_analyser.s

# target to generate assembly for a file
src/csvm/csvm_cluster_analyser.cc.s:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_cluster_analyser.cc.s
.PHONY : src/csvm/csvm_cluster_analyser.cc.s

src/csvm/csvm_codebook.o: src/csvm/csvm_codebook.cc.o
.PHONY : src/csvm/csvm_codebook.o

# target to build an object file
src/csvm/csvm_codebook.cc.o:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_codebook.cc.o
.PHONY : src/csvm/csvm_codebook.cc.o

src/csvm/csvm_codebook.i: src/csvm/csvm_codebook.cc.i
.PHONY : src/csvm/csvm_codebook.i

# target to preprocess a source file
src/csvm/csvm_codebook.cc.i:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_codebook.cc.i
.PHONY : src/csvm/csvm_codebook.cc.i

src/csvm/csvm_codebook.s: src/csvm/csvm_codebook.cc.s
.PHONY : src/csvm/csvm_codebook.s

# target to generate assembly for a file
src/csvm/csvm_codebook.cc.s:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_codebook.cc.s
.PHONY : src/csvm/csvm_codebook.cc.s

src/csvm/csvm_dataset.o: src/csvm/csvm_dataset.cc.o
.PHONY : src/csvm/csvm_dataset.o

# target to build an object file
src/csvm/csvm_dataset.cc.o:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_dataset.cc.o
.PHONY : src/csvm/csvm_dataset.cc.o

src/csvm/csvm_dataset.i: src/csvm/csvm_dataset.cc.i
.PHONY : src/csvm/csvm_dataset.i

# target to preprocess a source file
src/csvm/csvm_dataset.cc.i:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_dataset.cc.i
.PHONY : src/csvm/csvm_dataset.cc.i

src/csvm/csvm_dataset.s: src/csvm/csvm_dataset.cc.s
.PHONY : src/csvm/csvm_dataset.s

# target to generate assembly for a file
src/csvm/csvm_dataset.cc.s:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_dataset.cc.s
.PHONY : src/csvm/csvm_dataset.cc.s

src/csvm/csvm_feature.o: src/csvm/csvm_feature.cc.o
.PHONY : src/csvm/csvm_feature.o

# target to build an object file
src/csvm/csvm_feature.cc.o:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_feature.cc.o
.PHONY : src/csvm/csvm_feature.cc.o

src/csvm/csvm_feature.i: src/csvm/csvm_feature.cc.i
.PHONY : src/csvm/csvm_feature.i

# target to preprocess a source file
src/csvm/csvm_feature.cc.i:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_feature.cc.i
.PHONY : src/csvm/csvm_feature.cc.i

src/csvm/csvm_feature.s: src/csvm/csvm_feature.cc.s
.PHONY : src/csvm/csvm_feature.s

# target to generate assembly for a file
src/csvm/csvm_feature.cc.s:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_feature.cc.s
.PHONY : src/csvm/csvm_feature.cc.s

src/csvm/csvm_feature_extractor.o: src/csvm/csvm_feature_extractor.cc.o
.PHONY : src/csvm/csvm_feature_extractor.o

# target to build an object file
src/csvm/csvm_feature_extractor.cc.o:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_feature_extractor.cc.o
.PHONY : src/csvm/csvm_feature_extractor.cc.o

src/csvm/csvm_feature_extractor.i: src/csvm/csvm_feature_extractor.cc.i
.PHONY : src/csvm/csvm_feature_extractor.i

# target to preprocess a source file
src/csvm/csvm_feature_extractor.cc.i:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_feature_extractor.cc.i
.PHONY : src/csvm/csvm_feature_extractor.cc.i

src/csvm/csvm_feature_extractor.s: src/csvm/csvm_feature_extractor.cc.s
.PHONY : src/csvm/csvm_feature_extractor.s

# target to generate assembly for a file
src/csvm/csvm_feature_extractor.cc.s:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_feature_extractor.cc.s
.PHONY : src/csvm/csvm_feature_extractor.cc.s

src/csvm/csvm_frequency_matrix.o: src/csvm/csvm_frequency_matrix.cc.o
.PHONY : src/csvm/csvm_frequency_matrix.o

# target to build an object file
src/csvm/csvm_frequency_matrix.cc.o:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_frequency_matrix.cc.o
.PHONY : src/csvm/csvm_frequency_matrix.cc.o

src/csvm/csvm_frequency_matrix.i: src/csvm/csvm_frequency_matrix.cc.i
.PHONY : src/csvm/csvm_frequency_matrix.i

# target to preprocess a source file
src/csvm/csvm_frequency_matrix.cc.i:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_frequency_matrix.cc.i
.PHONY : src/csvm/csvm_frequency_matrix.cc.i

src/csvm/csvm_frequency_matrix.s: src/csvm/csvm_frequency_matrix.cc.s
.PHONY : src/csvm/csvm_frequency_matrix.s

# target to generate assembly for a file
src/csvm/csvm_frequency_matrix.cc.s:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_frequency_matrix.cc.s
.PHONY : src/csvm/csvm_frequency_matrix.cc.s

src/csvm/csvm_hog_descriptor.o: src/csvm/csvm_hog_descriptor.cc.o
.PHONY : src/csvm/csvm_hog_descriptor.o

# target to build an object file
src/csvm/csvm_hog_descriptor.cc.o:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_hog_descriptor.cc.o
.PHONY : src/csvm/csvm_hog_descriptor.cc.o

src/csvm/csvm_hog_descriptor.i: src/csvm/csvm_hog_descriptor.cc.i
.PHONY : src/csvm/csvm_hog_descriptor.i

# target to preprocess a source file
src/csvm/csvm_hog_descriptor.cc.i:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_hog_descriptor.cc.i
.PHONY : src/csvm/csvm_hog_descriptor.cc.i

src/csvm/csvm_hog_descriptor.s: src/csvm/csvm_hog_descriptor.cc.s
.PHONY : src/csvm/csvm_hog_descriptor.s

# target to generate assembly for a file
src/csvm/csvm_hog_descriptor.cc.s:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_hog_descriptor.cc.s
.PHONY : src/csvm/csvm_hog_descriptor.cc.s

src/csvm/csvm_image.o: src/csvm/csvm_image.cc.o
.PHONY : src/csvm/csvm_image.o

# target to build an object file
src/csvm/csvm_image.cc.o:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_image.cc.o
.PHONY : src/csvm/csvm_image.cc.o

src/csvm/csvm_image.i: src/csvm/csvm_image.cc.i
.PHONY : src/csvm/csvm_image.i

# target to preprocess a source file
src/csvm/csvm_image.cc.i:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_image.cc.i
.PHONY : src/csvm/csvm_image.cc.i

src/csvm/csvm_image.s: src/csvm/csvm_image.cc.s
.PHONY : src/csvm/csvm_image.s

# target to generate assembly for a file
src/csvm/csvm_image.cc.s:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_image.cc.s
.PHONY : src/csvm/csvm_image.cc.s

src/csvm/csvm_image_scanner.o: src/csvm/csvm_image_scanner.cc.o
.PHONY : src/csvm/csvm_image_scanner.o

# target to build an object file
src/csvm/csvm_image_scanner.cc.o:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_image_scanner.cc.o
.PHONY : src/csvm/csvm_image_scanner.cc.o

src/csvm/csvm_image_scanner.i: src/csvm/csvm_image_scanner.cc.i
.PHONY : src/csvm/csvm_image_scanner.i

# target to preprocess a source file
src/csvm/csvm_image_scanner.cc.i:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_image_scanner.cc.i
.PHONY : src/csvm/csvm_image_scanner.cc.i

src/csvm/csvm_image_scanner.s: src/csvm/csvm_image_scanner.cc.s
.PHONY : src/csvm/csvm_image_scanner.s

# target to generate assembly for a file
src/csvm/csvm_image_scanner.cc.s:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_image_scanner.cc.s
.PHONY : src/csvm/csvm_image_scanner.cc.s

src/csvm/csvm_kmeans.o: src/csvm/csvm_kmeans.cc.o
.PHONY : src/csvm/csvm_kmeans.o

# target to build an object file
src/csvm/csvm_kmeans.cc.o:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_kmeans.cc.o
.PHONY : src/csvm/csvm_kmeans.cc.o

src/csvm/csvm_kmeans.i: src/csvm/csvm_kmeans.cc.i
.PHONY : src/csvm/csvm_kmeans.i

# target to preprocess a source file
src/csvm/csvm_kmeans.cc.i:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_kmeans.cc.i
.PHONY : src/csvm/csvm_kmeans.cc.i

src/csvm/csvm_kmeans.s: src/csvm/csvm_kmeans.cc.s
.PHONY : src/csvm/csvm_kmeans.s

# target to generate assembly for a file
src/csvm/csvm_kmeans.cc.s:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_kmeans.cc.s
.PHONY : src/csvm/csvm_kmeans.cc.s

src/csvm/csvm_lbp_descriptor.o: src/csvm/csvm_lbp_descriptor.cc.o
.PHONY : src/csvm/csvm_lbp_descriptor.o

# target to build an object file
src/csvm/csvm_lbp_descriptor.cc.o:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_lbp_descriptor.cc.o
.PHONY : src/csvm/csvm_lbp_descriptor.cc.o

src/csvm/csvm_lbp_descriptor.i: src/csvm/csvm_lbp_descriptor.cc.i
.PHONY : src/csvm/csvm_lbp_descriptor.i

# target to preprocess a source file
src/csvm/csvm_lbp_descriptor.cc.i:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_lbp_descriptor.cc.i
.PHONY : src/csvm/csvm_lbp_descriptor.cc.i

src/csvm/csvm_lbp_descriptor.s: src/csvm/csvm_lbp_descriptor.cc.s
.PHONY : src/csvm/csvm_lbp_descriptor.s

# target to generate assembly for a file
src/csvm/csvm_lbp_descriptor.cc.s:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_lbp_descriptor.cc.s
.PHONY : src/csvm/csvm_lbp_descriptor.cc.s

src/csvm/csvm_lvq.o: src/csvm/csvm_lvq.cc.o
.PHONY : src/csvm/csvm_lvq.o

# target to build an object file
src/csvm/csvm_lvq.cc.o:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_lvq.cc.o
.PHONY : src/csvm/csvm_lvq.cc.o

src/csvm/csvm_lvq.i: src/csvm/csvm_lvq.cc.i
.PHONY : src/csvm/csvm_lvq.i

# target to preprocess a source file
src/csvm/csvm_lvq.cc.i:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_lvq.cc.i
.PHONY : src/csvm/csvm_lvq.cc.i

src/csvm/csvm_lvq.s: src/csvm/csvm_lvq.cc.s
.PHONY : src/csvm/csvm_lvq.s

# target to generate assembly for a file
src/csvm/csvm_lvq.cc.s:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_lvq.cc.s
.PHONY : src/csvm/csvm_lvq.cc.s

src/csvm/csvm_patch.o: src/csvm/csvm_patch.cc.o
.PHONY : src/csvm/csvm_patch.o

# target to build an object file
src/csvm/csvm_patch.cc.o:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_patch.cc.o
.PHONY : src/csvm/csvm_patch.cc.o

src/csvm/csvm_patch.i: src/csvm/csvm_patch.cc.i
.PHONY : src/csvm/csvm_patch.i

# target to preprocess a source file
src/csvm/csvm_patch.cc.i:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_patch.cc.i
.PHONY : src/csvm/csvm_patch.cc.i

src/csvm/csvm_patch.s: src/csvm/csvm_patch.cc.s
.PHONY : src/csvm/csvm_patch.s

# target to generate assembly for a file
src/csvm/csvm_patch.cc.s:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_patch.cc.s
.PHONY : src/csvm/csvm_patch.cc.s

src/csvm/csvm_rbm.o: src/csvm/csvm_rbm.cc.o
.PHONY : src/csvm/csvm_rbm.o

# target to build an object file
src/csvm/csvm_rbm.cc.o:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_rbm.cc.o
.PHONY : src/csvm/csvm_rbm.cc.o

src/csvm/csvm_rbm.i: src/csvm/csvm_rbm.cc.i
.PHONY : src/csvm/csvm_rbm.i

# target to preprocess a source file
src/csvm/csvm_rbm.cc.i:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_rbm.cc.i
.PHONY : src/csvm/csvm_rbm.cc.i

src/csvm/csvm_rbm.s: src/csvm/csvm_rbm.cc.s
.PHONY : src/csvm/csvm_rbm.s

# target to generate assembly for a file
src/csvm/csvm_rbm.cc.s:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_rbm.cc.s
.PHONY : src/csvm/csvm_rbm.cc.s

src/csvm/csvm_settings.o: src/csvm/csvm_settings.cc.o
.PHONY : src/csvm/csvm_settings.o

# target to build an object file
src/csvm/csvm_settings.cc.o:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_settings.cc.o
.PHONY : src/csvm/csvm_settings.cc.o

src/csvm/csvm_settings.i: src/csvm/csvm_settings.cc.i
.PHONY : src/csvm/csvm_settings.i

# target to preprocess a source file
src/csvm/csvm_settings.cc.i:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_settings.cc.i
.PHONY : src/csvm/csvm_settings.cc.i

src/csvm/csvm_settings.s: src/csvm/csvm_settings.cc.s
.PHONY : src/csvm/csvm_settings.s

# target to generate assembly for a file
src/csvm/csvm_settings.cc.s:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/csvm/csvm_settings.cc.s
.PHONY : src/csvm/csvm_settings.cc.s

src/dnn/dnn_data.o: src/dnn/dnn_data.c.o
.PHONY : src/dnn/dnn_data.o

# target to build an object file
src/dnn/dnn_data.c.o:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/dnn/dnn_data.c.o
.PHONY : src/dnn/dnn_data.c.o

src/dnn/dnn_data.i: src/dnn/dnn_data.c.i
.PHONY : src/dnn/dnn_data.i

# target to preprocess a source file
src/dnn/dnn_data.c.i:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/dnn/dnn_data.c.i
.PHONY : src/dnn/dnn_data.c.i

src/dnn/dnn_data.s: src/dnn/dnn_data.c.s
.PHONY : src/dnn/dnn_data.s

# target to generate assembly for a file
src/dnn/dnn_data.c.s:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/dnn/dnn_data.c.s
.PHONY : src/dnn/dnn_data.c.s

src/dnn/dnn_flow.o: src/dnn/dnn_flow.c.o
.PHONY : src/dnn/dnn_flow.o

# target to build an object file
src/dnn/dnn_flow.c.o:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/dnn/dnn_flow.c.o
.PHONY : src/dnn/dnn_flow.c.o

src/dnn/dnn_flow.i: src/dnn/dnn_flow.c.i
.PHONY : src/dnn/dnn_flow.i

# target to preprocess a source file
src/dnn/dnn_flow.c.i:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/dnn/dnn_flow.c.i
.PHONY : src/dnn/dnn_flow.c.i

src/dnn/dnn_flow.s: src/dnn/dnn_flow.c.s
.PHONY : src/dnn/dnn_flow.s

# target to generate assembly for a file
src/dnn/dnn_flow.c.s:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/dnn/dnn_flow.c.s
.PHONY : src/dnn/dnn_flow.c.s

src/dnn/dnn_layer_stack.o: src/dnn/dnn_layer_stack.c.o
.PHONY : src/dnn/dnn_layer_stack.o

# target to build an object file
src/dnn/dnn_layer_stack.c.o:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/dnn/dnn_layer_stack.c.o
.PHONY : src/dnn/dnn_layer_stack.c.o

src/dnn/dnn_layer_stack.i: src/dnn/dnn_layer_stack.c.i
.PHONY : src/dnn/dnn_layer_stack.i

# target to preprocess a source file
src/dnn/dnn_layer_stack.c.i:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/dnn/dnn_layer_stack.c.i
.PHONY : src/dnn/dnn_layer_stack.c.i

src/dnn/dnn_layer_stack.s: src/dnn/dnn_layer_stack.c.s
.PHONY : src/dnn/dnn_layer_stack.s

# target to generate assembly for a file
src/dnn/dnn_layer_stack.c.s:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/dnn/dnn_layer_stack.c.s
.PHONY : src/dnn/dnn_layer_stack.c.s

src/dnn/dnn_math.o: src/dnn/dnn_math.c.o
.PHONY : src/dnn/dnn_math.o

# target to build an object file
src/dnn/dnn_math.c.o:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/dnn/dnn_math.c.o
.PHONY : src/dnn/dnn_math.c.o

src/dnn/dnn_math.i: src/dnn/dnn_math.c.i
.PHONY : src/dnn/dnn_math.i

# target to preprocess a source file
src/dnn/dnn_math.c.i:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/dnn/dnn_math.c.i
.PHONY : src/dnn/dnn_math.c.i

src/dnn/dnn_math.s: src/dnn/dnn_math.c.s
.PHONY : src/dnn/dnn_math.s

# target to generate assembly for a file
src/dnn/dnn_math.c.s:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/dnn/dnn_math.c.s
.PHONY : src/dnn/dnn_math.c.s

src/dnn/dnn_pretrainer.o: src/dnn/dnn_pretrainer.c.o
.PHONY : src/dnn/dnn_pretrainer.o

# target to build an object file
src/dnn/dnn_pretrainer.c.o:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/dnn/dnn_pretrainer.c.o
.PHONY : src/dnn/dnn_pretrainer.c.o

src/dnn/dnn_pretrainer.i: src/dnn/dnn_pretrainer.c.i
.PHONY : src/dnn/dnn_pretrainer.i

# target to preprocess a source file
src/dnn/dnn_pretrainer.c.i:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/dnn/dnn_pretrainer.c.i
.PHONY : src/dnn/dnn_pretrainer.c.i

src/dnn/dnn_pretrainer.s: src/dnn/dnn_pretrainer.c.s
.PHONY : src/dnn/dnn_pretrainer.s

# target to generate assembly for a file
src/dnn/dnn_pretrainer.c.s:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/dnn/dnn_pretrainer.c.s
.PHONY : src/dnn/dnn_pretrainer.c.s

src/dnn/dnn_weights.o: src/dnn/dnn_weights.c.o
.PHONY : src/dnn/dnn_weights.o

# target to build an object file
src/dnn/dnn_weights.c.o:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/dnn/dnn_weights.c.o
.PHONY : src/dnn/dnn_weights.c.o

src/dnn/dnn_weights.i: src/dnn/dnn_weights.c.i
.PHONY : src/dnn/dnn_weights.i

# target to preprocess a source file
src/dnn/dnn_weights.c.i:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/dnn/dnn_weights.c.i
.PHONY : src/dnn/dnn_weights.c.i

src/dnn/dnn_weights.s: src/dnn/dnn_weights.c.s
.PHONY : src/dnn/dnn_weights.s

# target to generate assembly for a file
src/dnn/dnn_weights.c.s:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/dnn/dnn_weights.c.s
.PHONY : src/dnn/dnn_weights.c.s

src/lodepng/lodepng.o: src/lodepng/lodepng.cc.o
.PHONY : src/lodepng/lodepng.o

# target to build an object file
src/lodepng/lodepng.cc.o:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/lodepng/lodepng.cc.o
.PHONY : src/lodepng/lodepng.cc.o

src/lodepng/lodepng.i: src/lodepng/lodepng.cc.i
.PHONY : src/lodepng/lodepng.i

# target to preprocess a source file
src/lodepng/lodepng.cc.i:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/lodepng/lodepng.cc.i
.PHONY : src/lodepng/lodepng.cc.i

src/lodepng/lodepng.s: src/lodepng/lodepng.cc.s
.PHONY : src/lodepng/lodepng.s

# target to generate assembly for a file
src/lodepng/lodepng.cc.s:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/lodepng/lodepng.cc.s
.PHONY : src/lodepng/lodepng.cc.s

src/main.o: src/main.cc.o
.PHONY : src/main.o

# target to build an object file
src/main.cc.o:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/main.cc.o
.PHONY : src/main.cc.o

src/main.i: src/main.cc.i
.PHONY : src/main.i

# target to preprocess a source file
src/main.cc.i:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/main.cc.i
.PHONY : src/main.cc.i

src/main.s: src/main.cc.s
.PHONY : src/main.s

# target to generate assembly for a file
src/main.cc.s:
	$(MAKE) -f CMakeFiles/CSVM.dir/build.make CMakeFiles/CSVM.dir/src/main.cc.s
.PHONY : src/main.cc.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... CSVM"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... src/csvm/csvm_cifar10_parser.o"
	@echo "... src/csvm/csvm_cifar10_parser.i"
	@echo "... src/csvm/csvm_cifar10_parser.s"
	@echo "... src/csvm/csvm_classifier.o"
	@echo "... src/csvm/csvm_classifier.i"
	@echo "... src/csvm/csvm_classifier.s"
	@echo "... src/csvm/csvm_clean_descriptor.o"
	@echo "... src/csvm/csvm_clean_descriptor.i"
	@echo "... src/csvm/csvm_clean_descriptor.s"
	@echo "... src/csvm/csvm_cluster_analyser.o"
	@echo "... src/csvm/csvm_cluster_analyser.i"
	@echo "... src/csvm/csvm_cluster_analyser.s"
	@echo "... src/csvm/csvm_codebook.o"
	@echo "... src/csvm/csvm_codebook.i"
	@echo "... src/csvm/csvm_codebook.s"
	@echo "... src/csvm/csvm_dataset.o"
	@echo "... src/csvm/csvm_dataset.i"
	@echo "... src/csvm/csvm_dataset.s"
	@echo "... src/csvm/csvm_feature.o"
	@echo "... src/csvm/csvm_feature.i"
	@echo "... src/csvm/csvm_feature.s"
	@echo "... src/csvm/csvm_feature_extractor.o"
	@echo "... src/csvm/csvm_feature_extractor.i"
	@echo "... src/csvm/csvm_feature_extractor.s"
	@echo "... src/csvm/csvm_frequency_matrix.o"
	@echo "... src/csvm/csvm_frequency_matrix.i"
	@echo "... src/csvm/csvm_frequency_matrix.s"
	@echo "... src/csvm/csvm_hog_descriptor.o"
	@echo "... src/csvm/csvm_hog_descriptor.i"
	@echo "... src/csvm/csvm_hog_descriptor.s"
	@echo "... src/csvm/csvm_image.o"
	@echo "... src/csvm/csvm_image.i"
	@echo "... src/csvm/csvm_image.s"
	@echo "... src/csvm/csvm_image_scanner.o"
	@echo "... src/csvm/csvm_image_scanner.i"
	@echo "... src/csvm/csvm_image_scanner.s"
	@echo "... src/csvm/csvm_kmeans.o"
	@echo "... src/csvm/csvm_kmeans.i"
	@echo "... src/csvm/csvm_kmeans.s"
	@echo "... src/csvm/csvm_lbp_descriptor.o"
	@echo "... src/csvm/csvm_lbp_descriptor.i"
	@echo "... src/csvm/csvm_lbp_descriptor.s"
	@echo "... src/csvm/csvm_lvq.o"
	@echo "... src/csvm/csvm_lvq.i"
	@echo "... src/csvm/csvm_lvq.s"
	@echo "... src/csvm/csvm_patch.o"
	@echo "... src/csvm/csvm_patch.i"
	@echo "... src/csvm/csvm_patch.s"
	@echo "... src/csvm/csvm_rbm.o"
	@echo "... src/csvm/csvm_rbm.i"
	@echo "... src/csvm/csvm_rbm.s"
	@echo "... src/csvm/csvm_settings.o"
	@echo "... src/csvm/csvm_settings.i"
	@echo "... src/csvm/csvm_settings.s"
	@echo "... src/dnn/dnn_data.o"
	@echo "... src/dnn/dnn_data.i"
	@echo "... src/dnn/dnn_data.s"
	@echo "... src/dnn/dnn_flow.o"
	@echo "... src/dnn/dnn_flow.i"
	@echo "... src/dnn/dnn_flow.s"
	@echo "... src/dnn/dnn_layer_stack.o"
	@echo "... src/dnn/dnn_layer_stack.i"
	@echo "... src/dnn/dnn_layer_stack.s"
	@echo "... src/dnn/dnn_math.o"
	@echo "... src/dnn/dnn_math.i"
	@echo "... src/dnn/dnn_math.s"
	@echo "... src/dnn/dnn_pretrainer.o"
	@echo "... src/dnn/dnn_pretrainer.i"
	@echo "... src/dnn/dnn_pretrainer.s"
	@echo "... src/dnn/dnn_weights.o"
	@echo "... src/dnn/dnn_weights.i"
	@echo "... src/dnn/dnn_weights.s"
	@echo "... src/lodepng/lodepng.o"
	@echo "... src/lodepng/lodepng.i"
	@echo "... src/lodepng/lodepng.s"
	@echo "... src/main.o"
	@echo "... src/main.i"
	@echo "... src/main.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

