# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.14

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


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
CMAKE_COMMAND = /opt/miniconda3/bin/cmake

# The command to remove a file.
RM = /opt/miniconda3/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /workspace/TensorRT/demo/BERT

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /workspace/TensorRT/demo/BERT/build2

# Include any dependencies generated for this target.
include CMakeFiles/sample_bert_model.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/sample_bert_model.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/sample_bert_model.dir/flags.make

CMakeFiles/sample_bert_model.dir/sampleBERT-model.cpp.o: CMakeFiles/sample_bert_model.dir/flags.make
CMakeFiles/sample_bert_model.dir/sampleBERT-model.cpp.o: ../sampleBERT-model.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/TensorRT/demo/BERT/build2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/sample_bert_model.dir/sampleBERT-model.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/sample_bert_model.dir/sampleBERT-model.cpp.o -c /workspace/TensorRT/demo/BERT/sampleBERT-model.cpp

CMakeFiles/sample_bert_model.dir/sampleBERT-model.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/sample_bert_model.dir/sampleBERT-model.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /workspace/TensorRT/demo/BERT/sampleBERT-model.cpp > CMakeFiles/sample_bert_model.dir/sampleBERT-model.cpp.i

CMakeFiles/sample_bert_model.dir/sampleBERT-model.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/sample_bert_model.dir/sampleBERT-model.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /workspace/TensorRT/demo/BERT/sampleBERT-model.cpp -o CMakeFiles/sample_bert_model.dir/sampleBERT-model.cpp.s

# Object files for target sample_bert_model
sample_bert_model_OBJECTS = \
"CMakeFiles/sample_bert_model.dir/sampleBERT-model.cpp.o"

# External object files for target sample_bert_model
sample_bert_model_EXTERNAL_OBJECTS =

sample_bert_model: CMakeFiles/sample_bert_model.dir/sampleBERT-model.cpp.o
sample_bert_model: CMakeFiles/sample_bert_model.dir/build.make
sample_bert_model: libcommon.so
sample_bert_model: libbert_plugins.so
sample_bert_model: CMakeFiles/sample_bert_model.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/TensorRT/demo/BERT/build2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable sample_bert_model"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/sample_bert_model.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/sample_bert_model.dir/build: sample_bert_model

.PHONY : CMakeFiles/sample_bert_model.dir/build

CMakeFiles/sample_bert_model.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/sample_bert_model.dir/cmake_clean.cmake
.PHONY : CMakeFiles/sample_bert_model.dir/clean

CMakeFiles/sample_bert_model.dir/depend:
	cd /workspace/TensorRT/demo/BERT/build2 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/TensorRT/demo/BERT /workspace/TensorRT/demo/BERT /workspace/TensorRT/demo/BERT/build2 /workspace/TensorRT/demo/BERT/build2 /workspace/TensorRT/demo/BERT/build2/CMakeFiles/sample_bert_model.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/sample_bert_model.dir/depend

