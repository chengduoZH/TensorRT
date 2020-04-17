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
include CMakeFiles/bert_plugins.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/bert_plugins.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/bert_plugins.dir/flags.make

CMakeFiles/bert_plugins.dir/plugins/geluPlugin.cu.o: CMakeFiles/bert_plugins.dir/flags.make
CMakeFiles/bert_plugins.dir/plugins/geluPlugin.cu.o: ../plugins/geluPlugin.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/TensorRT/demo/BERT/build2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object CMakeFiles/bert_plugins.dir/plugins/geluPlugin.cu.o"
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /workspace/TensorRT/demo/BERT/plugins/geluPlugin.cu -o CMakeFiles/bert_plugins.dir/plugins/geluPlugin.cu.o

CMakeFiles/bert_plugins.dir/plugins/geluPlugin.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/bert_plugins.dir/plugins/geluPlugin.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/bert_plugins.dir/plugins/geluPlugin.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/bert_plugins.dir/plugins/geluPlugin.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/bert_plugins.dir/plugins/skipLayerNormPlugin.cu.o: CMakeFiles/bert_plugins.dir/flags.make
CMakeFiles/bert_plugins.dir/plugins/skipLayerNormPlugin.cu.o: ../plugins/skipLayerNormPlugin.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/TensorRT/demo/BERT/build2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CUDA object CMakeFiles/bert_plugins.dir/plugins/skipLayerNormPlugin.cu.o"
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /workspace/TensorRT/demo/BERT/plugins/skipLayerNormPlugin.cu -o CMakeFiles/bert_plugins.dir/plugins/skipLayerNormPlugin.cu.o

CMakeFiles/bert_plugins.dir/plugins/skipLayerNormPlugin.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/bert_plugins.dir/plugins/skipLayerNormPlugin.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/bert_plugins.dir/plugins/skipLayerNormPlugin.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/bert_plugins.dir/plugins/skipLayerNormPlugin.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/bert_plugins.dir/plugins/qkvToContextPlugin.cu.o: CMakeFiles/bert_plugins.dir/flags.make
CMakeFiles/bert_plugins.dir/plugins/qkvToContextPlugin.cu.o: ../plugins/qkvToContextPlugin.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/TensorRT/demo/BERT/build2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CUDA object CMakeFiles/bert_plugins.dir/plugins/qkvToContextPlugin.cu.o"
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /workspace/TensorRT/demo/BERT/plugins/qkvToContextPlugin.cu -o CMakeFiles/bert_plugins.dir/plugins/qkvToContextPlugin.cu.o

CMakeFiles/bert_plugins.dir/plugins/qkvToContextPlugin.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/bert_plugins.dir/plugins/qkvToContextPlugin.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/bert_plugins.dir/plugins/qkvToContextPlugin.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/bert_plugins.dir/plugins/qkvToContextPlugin.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

CMakeFiles/bert_plugins.dir/plugins/embLayerNormPlugin.cu.o: CMakeFiles/bert_plugins.dir/flags.make
CMakeFiles/bert_plugins.dir/plugins/embLayerNormPlugin.cu.o: ../plugins/embLayerNormPlugin.cu
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/workspace/TensorRT/demo/BERT/build2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CUDA object CMakeFiles/bert_plugins.dir/plugins/embLayerNormPlugin.cu.o"
	/usr/local/cuda/bin/nvcc  $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -x cu -c /workspace/TensorRT/demo/BERT/plugins/embLayerNormPlugin.cu -o CMakeFiles/bert_plugins.dir/plugins/embLayerNormPlugin.cu.o

CMakeFiles/bert_plugins.dir/plugins/embLayerNormPlugin.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CUDA source to CMakeFiles/bert_plugins.dir/plugins/embLayerNormPlugin.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

CMakeFiles/bert_plugins.dir/plugins/embLayerNormPlugin.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CUDA source to assembly CMakeFiles/bert_plugins.dir/plugins/embLayerNormPlugin.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target bert_plugins
bert_plugins_OBJECTS = \
"CMakeFiles/bert_plugins.dir/plugins/geluPlugin.cu.o" \
"CMakeFiles/bert_plugins.dir/plugins/skipLayerNormPlugin.cu.o" \
"CMakeFiles/bert_plugins.dir/plugins/qkvToContextPlugin.cu.o" \
"CMakeFiles/bert_plugins.dir/plugins/embLayerNormPlugin.cu.o"

# External object files for target bert_plugins
bert_plugins_EXTERNAL_OBJECTS =

CMakeFiles/bert_plugins.dir/cmake_device_link.o: CMakeFiles/bert_plugins.dir/plugins/geluPlugin.cu.o
CMakeFiles/bert_plugins.dir/cmake_device_link.o: CMakeFiles/bert_plugins.dir/plugins/skipLayerNormPlugin.cu.o
CMakeFiles/bert_plugins.dir/cmake_device_link.o: CMakeFiles/bert_plugins.dir/plugins/qkvToContextPlugin.cu.o
CMakeFiles/bert_plugins.dir/cmake_device_link.o: CMakeFiles/bert_plugins.dir/plugins/embLayerNormPlugin.cu.o
CMakeFiles/bert_plugins.dir/cmake_device_link.o: CMakeFiles/bert_plugins.dir/build.make
CMakeFiles/bert_plugins.dir/cmake_device_link.o: CMakeFiles/bert_plugins.dir/dlink.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/TensorRT/demo/BERT/build2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Linking CUDA device code CMakeFiles/bert_plugins.dir/cmake_device_link.o"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bert_plugins.dir/dlink.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/bert_plugins.dir/build: CMakeFiles/bert_plugins.dir/cmake_device_link.o

.PHONY : CMakeFiles/bert_plugins.dir/build

# Object files for target bert_plugins
bert_plugins_OBJECTS = \
"CMakeFiles/bert_plugins.dir/plugins/geluPlugin.cu.o" \
"CMakeFiles/bert_plugins.dir/plugins/skipLayerNormPlugin.cu.o" \
"CMakeFiles/bert_plugins.dir/plugins/qkvToContextPlugin.cu.o" \
"CMakeFiles/bert_plugins.dir/plugins/embLayerNormPlugin.cu.o"

# External object files for target bert_plugins
bert_plugins_EXTERNAL_OBJECTS =

libbert_plugins.so: CMakeFiles/bert_plugins.dir/plugins/geluPlugin.cu.o
libbert_plugins.so: CMakeFiles/bert_plugins.dir/plugins/skipLayerNormPlugin.cu.o
libbert_plugins.so: CMakeFiles/bert_plugins.dir/plugins/qkvToContextPlugin.cu.o
libbert_plugins.so: CMakeFiles/bert_plugins.dir/plugins/embLayerNormPlugin.cu.o
libbert_plugins.so: CMakeFiles/bert_plugins.dir/build.make
libbert_plugins.so: CMakeFiles/bert_plugins.dir/cmake_device_link.o
libbert_plugins.so: CMakeFiles/bert_plugins.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/workspace/TensorRT/demo/BERT/build2/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CUDA shared library libbert_plugins.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/bert_plugins.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/bert_plugins.dir/build: libbert_plugins.so

.PHONY : CMakeFiles/bert_plugins.dir/build

CMakeFiles/bert_plugins.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/bert_plugins.dir/cmake_clean.cmake
.PHONY : CMakeFiles/bert_plugins.dir/clean

CMakeFiles/bert_plugins.dir/depend:
	cd /workspace/TensorRT/demo/BERT/build2 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /workspace/TensorRT/demo/BERT /workspace/TensorRT/demo/BERT /workspace/TensorRT/demo/BERT/build2 /workspace/TensorRT/demo/BERT/build2 /workspace/TensorRT/demo/BERT/build2/CMakeFiles/bert_plugins.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/bert_plugins.dir/depend

