# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/qwe/anaconda3/lib/python3.9/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/qwe/anaconda3/lib/python3.9/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/qwe/code/ocr

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/qwe/code/ocr/build

# Include any dependencies generated for this target.
include CMakeFiles/ocrlib.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/ocrlib.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/ocrlib.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/ocrlib.dir/flags.make

CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_reader.cc.o: CMakeFiles/ocrlib.dir/flags.make
CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_reader.cc.o: ../3rdparty/jsoncpp/json_reader.cc
CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_reader.cc.o: CMakeFiles/ocrlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qwe/code/ocr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_reader.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_reader.cc.o -MF CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_reader.cc.o.d -o CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_reader.cc.o -c /home/qwe/code/ocr/3rdparty/jsoncpp/json_reader.cc

CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_reader.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_reader.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qwe/code/ocr/3rdparty/jsoncpp/json_reader.cc > CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_reader.cc.i

CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_reader.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_reader.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qwe/code/ocr/3rdparty/jsoncpp/json_reader.cc -o CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_reader.cc.s

CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_value.cc.o: CMakeFiles/ocrlib.dir/flags.make
CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_value.cc.o: ../3rdparty/jsoncpp/json_value.cc
CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_value.cc.o: CMakeFiles/ocrlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qwe/code/ocr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_value.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_value.cc.o -MF CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_value.cc.o.d -o CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_value.cc.o -c /home/qwe/code/ocr/3rdparty/jsoncpp/json_value.cc

CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_value.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_value.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qwe/code/ocr/3rdparty/jsoncpp/json_value.cc > CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_value.cc.i

CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_value.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_value.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qwe/code/ocr/3rdparty/jsoncpp/json_value.cc -o CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_value.cc.s

CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_writer.cc.o: CMakeFiles/ocrlib.dir/flags.make
CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_writer.cc.o: ../3rdparty/jsoncpp/json_writer.cc
CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_writer.cc.o: CMakeFiles/ocrlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qwe/code/ocr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_writer.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_writer.cc.o -MF CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_writer.cc.o.d -o CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_writer.cc.o -c /home/qwe/code/ocr/3rdparty/jsoncpp/json_writer.cc

CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_writer.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_writer.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qwe/code/ocr/3rdparty/jsoncpp/json_writer.cc > CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_writer.cc.i

CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_writer.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_writer.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qwe/code/ocr/3rdparty/jsoncpp/json_writer.cc -o CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_writer.cc.s

CMakeFiles/ocrlib.dir/ocr/cls.cc.o: CMakeFiles/ocrlib.dir/flags.make
CMakeFiles/ocrlib.dir/ocr/cls.cc.o: ../ocr/cls.cc
CMakeFiles/ocrlib.dir/ocr/cls.cc.o: CMakeFiles/ocrlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qwe/code/ocr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/ocrlib.dir/ocr/cls.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ocrlib.dir/ocr/cls.cc.o -MF CMakeFiles/ocrlib.dir/ocr/cls.cc.o.d -o CMakeFiles/ocrlib.dir/ocr/cls.cc.o -c /home/qwe/code/ocr/ocr/cls.cc

CMakeFiles/ocrlib.dir/ocr/cls.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ocrlib.dir/ocr/cls.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qwe/code/ocr/ocr/cls.cc > CMakeFiles/ocrlib.dir/ocr/cls.cc.i

CMakeFiles/ocrlib.dir/ocr/cls.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ocrlib.dir/ocr/cls.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qwe/code/ocr/ocr/cls.cc -o CMakeFiles/ocrlib.dir/ocr/cls.cc.s

CMakeFiles/ocrlib.dir/ocr/det.cc.o: CMakeFiles/ocrlib.dir/flags.make
CMakeFiles/ocrlib.dir/ocr/det.cc.o: ../ocr/det.cc
CMakeFiles/ocrlib.dir/ocr/det.cc.o: CMakeFiles/ocrlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qwe/code/ocr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/ocrlib.dir/ocr/det.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ocrlib.dir/ocr/det.cc.o -MF CMakeFiles/ocrlib.dir/ocr/det.cc.o.d -o CMakeFiles/ocrlib.dir/ocr/det.cc.o -c /home/qwe/code/ocr/ocr/det.cc

CMakeFiles/ocrlib.dir/ocr/det.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ocrlib.dir/ocr/det.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qwe/code/ocr/ocr/det.cc > CMakeFiles/ocrlib.dir/ocr/det.cc.i

CMakeFiles/ocrlib.dir/ocr/det.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ocrlib.dir/ocr/det.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qwe/code/ocr/ocr/det.cc -o CMakeFiles/ocrlib.dir/ocr/det.cc.s

CMakeFiles/ocrlib.dir/ocr/inference.cc.o: CMakeFiles/ocrlib.dir/flags.make
CMakeFiles/ocrlib.dir/ocr/inference.cc.o: ../ocr/inference.cc
CMakeFiles/ocrlib.dir/ocr/inference.cc.o: CMakeFiles/ocrlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qwe/code/ocr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/ocrlib.dir/ocr/inference.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ocrlib.dir/ocr/inference.cc.o -MF CMakeFiles/ocrlib.dir/ocr/inference.cc.o.d -o CMakeFiles/ocrlib.dir/ocr/inference.cc.o -c /home/qwe/code/ocr/ocr/inference.cc

CMakeFiles/ocrlib.dir/ocr/inference.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ocrlib.dir/ocr/inference.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qwe/code/ocr/ocr/inference.cc > CMakeFiles/ocrlib.dir/ocr/inference.cc.i

CMakeFiles/ocrlib.dir/ocr/inference.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ocrlib.dir/ocr/inference.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qwe/code/ocr/ocr/inference.cc -o CMakeFiles/ocrlib.dir/ocr/inference.cc.s

CMakeFiles/ocrlib.dir/ocr/inference_impl.cc.o: CMakeFiles/ocrlib.dir/flags.make
CMakeFiles/ocrlib.dir/ocr/inference_impl.cc.o: ../ocr/inference_impl.cc
CMakeFiles/ocrlib.dir/ocr/inference_impl.cc.o: CMakeFiles/ocrlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qwe/code/ocr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Building CXX object CMakeFiles/ocrlib.dir/ocr/inference_impl.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ocrlib.dir/ocr/inference_impl.cc.o -MF CMakeFiles/ocrlib.dir/ocr/inference_impl.cc.o.d -o CMakeFiles/ocrlib.dir/ocr/inference_impl.cc.o -c /home/qwe/code/ocr/ocr/inference_impl.cc

CMakeFiles/ocrlib.dir/ocr/inference_impl.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ocrlib.dir/ocr/inference_impl.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qwe/code/ocr/ocr/inference_impl.cc > CMakeFiles/ocrlib.dir/ocr/inference_impl.cc.i

CMakeFiles/ocrlib.dir/ocr/inference_impl.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ocrlib.dir/ocr/inference_impl.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qwe/code/ocr/ocr/inference_impl.cc -o CMakeFiles/ocrlib.dir/ocr/inference_impl.cc.s

CMakeFiles/ocrlib.dir/ocr/rec.cc.o: CMakeFiles/ocrlib.dir/flags.make
CMakeFiles/ocrlib.dir/ocr/rec.cc.o: ../ocr/rec.cc
CMakeFiles/ocrlib.dir/ocr/rec.cc.o: CMakeFiles/ocrlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qwe/code/ocr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "Building CXX object CMakeFiles/ocrlib.dir/ocr/rec.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ocrlib.dir/ocr/rec.cc.o -MF CMakeFiles/ocrlib.dir/ocr/rec.cc.o.d -o CMakeFiles/ocrlib.dir/ocr/rec.cc.o -c /home/qwe/code/ocr/ocr/rec.cc

CMakeFiles/ocrlib.dir/ocr/rec.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ocrlib.dir/ocr/rec.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qwe/code/ocr/ocr/rec.cc > CMakeFiles/ocrlib.dir/ocr/rec.cc.i

CMakeFiles/ocrlib.dir/ocr/rec.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ocrlib.dir/ocr/rec.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qwe/code/ocr/ocr/rec.cc -o CMakeFiles/ocrlib.dir/ocr/rec.cc.s

CMakeFiles/ocrlib.dir/ocr/tool/clipper.cc.o: CMakeFiles/ocrlib.dir/flags.make
CMakeFiles/ocrlib.dir/ocr/tool/clipper.cc.o: ../ocr/tool/clipper.cc
CMakeFiles/ocrlib.dir/ocr/tool/clipper.cc.o: CMakeFiles/ocrlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qwe/code/ocr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "Building CXX object CMakeFiles/ocrlib.dir/ocr/tool/clipper.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ocrlib.dir/ocr/tool/clipper.cc.o -MF CMakeFiles/ocrlib.dir/ocr/tool/clipper.cc.o.d -o CMakeFiles/ocrlib.dir/ocr/tool/clipper.cc.o -c /home/qwe/code/ocr/ocr/tool/clipper.cc

CMakeFiles/ocrlib.dir/ocr/tool/clipper.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ocrlib.dir/ocr/tool/clipper.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qwe/code/ocr/ocr/tool/clipper.cc > CMakeFiles/ocrlib.dir/ocr/tool/clipper.cc.i

CMakeFiles/ocrlib.dir/ocr/tool/clipper.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ocrlib.dir/ocr/tool/clipper.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qwe/code/ocr/ocr/tool/clipper.cc -o CMakeFiles/ocrlib.dir/ocr/tool/clipper.cc.s

CMakeFiles/ocrlib.dir/ocr/tool/postprocess_op.cc.o: CMakeFiles/ocrlib.dir/flags.make
CMakeFiles/ocrlib.dir/ocr/tool/postprocess_op.cc.o: ../ocr/tool/postprocess_op.cc
CMakeFiles/ocrlib.dir/ocr/tool/postprocess_op.cc.o: CMakeFiles/ocrlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qwe/code/ocr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "Building CXX object CMakeFiles/ocrlib.dir/ocr/tool/postprocess_op.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ocrlib.dir/ocr/tool/postprocess_op.cc.o -MF CMakeFiles/ocrlib.dir/ocr/tool/postprocess_op.cc.o.d -o CMakeFiles/ocrlib.dir/ocr/tool/postprocess_op.cc.o -c /home/qwe/code/ocr/ocr/tool/postprocess_op.cc

CMakeFiles/ocrlib.dir/ocr/tool/postprocess_op.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ocrlib.dir/ocr/tool/postprocess_op.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qwe/code/ocr/ocr/tool/postprocess_op.cc > CMakeFiles/ocrlib.dir/ocr/tool/postprocess_op.cc.i

CMakeFiles/ocrlib.dir/ocr/tool/postprocess_op.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ocrlib.dir/ocr/tool/postprocess_op.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qwe/code/ocr/ocr/tool/postprocess_op.cc -o CMakeFiles/ocrlib.dir/ocr/tool/postprocess_op.cc.s

CMakeFiles/ocrlib.dir/ocr/tool/util.cc.o: CMakeFiles/ocrlib.dir/flags.make
CMakeFiles/ocrlib.dir/ocr/tool/util.cc.o: ../ocr/tool/util.cc
CMakeFiles/ocrlib.dir/ocr/tool/util.cc.o: CMakeFiles/ocrlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qwe/code/ocr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_11) "Building CXX object CMakeFiles/ocrlib.dir/ocr/tool/util.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ocrlib.dir/ocr/tool/util.cc.o -MF CMakeFiles/ocrlib.dir/ocr/tool/util.cc.o.d -o CMakeFiles/ocrlib.dir/ocr/tool/util.cc.o -c /home/qwe/code/ocr/ocr/tool/util.cc

CMakeFiles/ocrlib.dir/ocr/tool/util.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ocrlib.dir/ocr/tool/util.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qwe/code/ocr/ocr/tool/util.cc > CMakeFiles/ocrlib.dir/ocr/tool/util.cc.i

CMakeFiles/ocrlib.dir/ocr/tool/util.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ocrlib.dir/ocr/tool/util.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qwe/code/ocr/ocr/tool/util.cc -o CMakeFiles/ocrlib.dir/ocr/tool/util.cc.s

CMakeFiles/ocrlib.dir/ocr/tool/utility.cc.o: CMakeFiles/ocrlib.dir/flags.make
CMakeFiles/ocrlib.dir/ocr/tool/utility.cc.o: ../ocr/tool/utility.cc
CMakeFiles/ocrlib.dir/ocr/tool/utility.cc.o: CMakeFiles/ocrlib.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/qwe/code/ocr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_12) "Building CXX object CMakeFiles/ocrlib.dir/ocr/tool/utility.cc.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/ocrlib.dir/ocr/tool/utility.cc.o -MF CMakeFiles/ocrlib.dir/ocr/tool/utility.cc.o.d -o CMakeFiles/ocrlib.dir/ocr/tool/utility.cc.o -c /home/qwe/code/ocr/ocr/tool/utility.cc

CMakeFiles/ocrlib.dir/ocr/tool/utility.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/ocrlib.dir/ocr/tool/utility.cc.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/qwe/code/ocr/ocr/tool/utility.cc > CMakeFiles/ocrlib.dir/ocr/tool/utility.cc.i

CMakeFiles/ocrlib.dir/ocr/tool/utility.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/ocrlib.dir/ocr/tool/utility.cc.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/qwe/code/ocr/ocr/tool/utility.cc -o CMakeFiles/ocrlib.dir/ocr/tool/utility.cc.s

# Object files for target ocrlib
ocrlib_OBJECTS = \
"CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_reader.cc.o" \
"CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_value.cc.o" \
"CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_writer.cc.o" \
"CMakeFiles/ocrlib.dir/ocr/cls.cc.o" \
"CMakeFiles/ocrlib.dir/ocr/det.cc.o" \
"CMakeFiles/ocrlib.dir/ocr/inference.cc.o" \
"CMakeFiles/ocrlib.dir/ocr/inference_impl.cc.o" \
"CMakeFiles/ocrlib.dir/ocr/rec.cc.o" \
"CMakeFiles/ocrlib.dir/ocr/tool/clipper.cc.o" \
"CMakeFiles/ocrlib.dir/ocr/tool/postprocess_op.cc.o" \
"CMakeFiles/ocrlib.dir/ocr/tool/util.cc.o" \
"CMakeFiles/ocrlib.dir/ocr/tool/utility.cc.o"

# External object files for target ocrlib
ocrlib_EXTERNAL_OBJECTS =

lib/libocrlib.so: CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_reader.cc.o
lib/libocrlib.so: CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_value.cc.o
lib/libocrlib.so: CMakeFiles/ocrlib.dir/3rdparty/jsoncpp/json_writer.cc.o
lib/libocrlib.so: CMakeFiles/ocrlib.dir/ocr/cls.cc.o
lib/libocrlib.so: CMakeFiles/ocrlib.dir/ocr/det.cc.o
lib/libocrlib.so: CMakeFiles/ocrlib.dir/ocr/inference.cc.o
lib/libocrlib.so: CMakeFiles/ocrlib.dir/ocr/inference_impl.cc.o
lib/libocrlib.so: CMakeFiles/ocrlib.dir/ocr/rec.cc.o
lib/libocrlib.so: CMakeFiles/ocrlib.dir/ocr/tool/clipper.cc.o
lib/libocrlib.so: CMakeFiles/ocrlib.dir/ocr/tool/postprocess_op.cc.o
lib/libocrlib.so: CMakeFiles/ocrlib.dir/ocr/tool/util.cc.o
lib/libocrlib.so: CMakeFiles/ocrlib.dir/ocr/tool/utility.cc.o
lib/libocrlib.so: CMakeFiles/ocrlib.dir/build.make
lib/libocrlib.so: /usr/local/lib/libopencv_stitching.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_superres.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_videostab.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_aruco.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_bgsegm.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_bioinspired.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_ccalib.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_dnn_objdetect.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_dpm.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_face.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_freetype.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_fuzzy.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_hfs.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_img_hash.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_line_descriptor.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_optflow.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_reg.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_rgbd.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_saliency.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_stereo.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_structured_light.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_surface_matching.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_tracking.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_xfeatures2d.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_ximgproc.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_xobjdetect.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_xphoto.so.3.4.3
lib/libocrlib.so: ../3rdparty/ncnn/x86/lib/libncnn.so.1
lib/libocrlib.so: /usr/local/lib/libopencv_shape.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_photo.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_calib3d.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_phase_unwrapping.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_video.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_datasets.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_plot.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_text.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_dnn.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_features2d.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_flann.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_highgui.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_ml.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_videoio.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_imgcodecs.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_objdetect.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_imgproc.so.3.4.3
lib/libocrlib.so: /usr/local/lib/libopencv_core.so.3.4.3
lib/libocrlib.so: CMakeFiles/ocrlib.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/qwe/code/ocr/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_13) "Linking CXX shared library lib/libocrlib.so"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/ocrlib.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/ocrlib.dir/build: lib/libocrlib.so
.PHONY : CMakeFiles/ocrlib.dir/build

CMakeFiles/ocrlib.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/ocrlib.dir/cmake_clean.cmake
.PHONY : CMakeFiles/ocrlib.dir/clean

CMakeFiles/ocrlib.dir/depend:
	cd /home/qwe/code/ocr/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/qwe/code/ocr /home/qwe/code/ocr /home/qwe/code/ocr/build /home/qwe/code/ocr/build /home/qwe/code/ocr/build/CMakeFiles/ocrlib.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/ocrlib.dir/depend

