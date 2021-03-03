# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.18

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
CMAKE_COMMAND = /usr/local/lib/python3.6/dist-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /usr/local/lib/python3.6/dist-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/manu/robocomp/components/ProyectoMM/Sistemas-Multimedia/ComponentSM

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/manu/robocomp/components/ProyectoMM/Sistemas-Multimedia/ComponentSM

# Utility rule file for ICES__home_manu_robocomp_components_ProyectoMM_Sistemas-Multimedia_ComponentSM_src.

# Include the progress variables for this target.
include src/CMakeFiles/ICES__home_manu_robocomp_components_ProyectoMM_Sistemas-Multimedia_ComponentSM_src.dir/progress.make

ICES__home_manu_robocomp_components_ProyectoMM_Sistemas-Multimedia_ComponentSM_src: src/CMakeFiles/ICES__home_manu_robocomp_components_ProyectoMM_Sistemas-Multimedia_ComponentSM_src.dir/build.make
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Generating CommonBehavior.ice from /home/manu/robocomp/interfaces/IDSLs/CommonBehavior.idsl"
	cd /home/manu/robocomp/components/ProyectoMM/Sistemas-Multimedia/ComponentSM/src && robocompdsl /home/manu/robocomp/interfaces/IDSLs/CommonBehavior.idsl /home/manu/robocomp/components/ProyectoMM/Sistemas-Multimedia/ComponentSM/src/CommonBehavior.ice
	cd /home/manu/robocomp/components/ProyectoMM/Sistemas-Multimedia/ComponentSM/src && robocompdsl /home/manu/robocomp/interfaces/IDSLs/CommonBehavior.idsl CommonBehavior.ice
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Generating CameraRGBDSimple.ice from /home/manu/robocomp/interfaces/IDSLs/CameraRGBDSimple.idsl"
	cd /home/manu/robocomp/components/ProyectoMM/Sistemas-Multimedia/ComponentSM/src && robocompdsl /home/manu/robocomp/interfaces/IDSLs/CameraRGBDSimple.idsl /home/manu/robocomp/components/ProyectoMM/Sistemas-Multimedia/ComponentSM/src/CameraRGBDSimple.ice
	cd /home/manu/robocomp/components/ProyectoMM/Sistemas-Multimedia/ComponentSM/src && robocompdsl /home/manu/robocomp/interfaces/IDSLs/CameraRGBDSimple.idsl CameraRGBDSimple.ice
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Generating CameraRGBDSimplePub.ice from /home/manu/robocomp/interfaces/IDSLs/CameraRGBDSimplePub.idsl"
	cd /home/manu/robocomp/components/ProyectoMM/Sistemas-Multimedia/ComponentSM/src && robocompdsl /home/manu/robocomp/interfaces/IDSLs/CameraRGBDSimplePub.idsl /home/manu/robocomp/components/ProyectoMM/Sistemas-Multimedia/ComponentSM/src/CameraRGBDSimplePub.ice
	cd /home/manu/robocomp/components/ProyectoMM/Sistemas-Multimedia/ComponentSM/src && robocompdsl /home/manu/robocomp/interfaces/IDSLs/CameraRGBDSimplePub.idsl CameraRGBDSimplePub.ice
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold "Generating HumanCameraBody.ice from /home/manu/robocomp/interfaces/IDSLs/HumanCameraBody.idsl"
	cd /home/manu/robocomp/components/ProyectoMM/Sistemas-Multimedia/ComponentSM/src && robocompdsl /home/manu/robocomp/interfaces/IDSLs/HumanCameraBody.idsl /home/manu/robocomp/components/ProyectoMM/Sistemas-Multimedia/ComponentSM/src/HumanCameraBody.ice
	cd /home/manu/robocomp/components/ProyectoMM/Sistemas-Multimedia/ComponentSM/src && robocompdsl /home/manu/robocomp/interfaces/IDSLs/HumanCameraBody.idsl HumanCameraBody.ice
.PHONY : ICES__home_manu_robocomp_components_ProyectoMM_Sistemas-Multimedia_ComponentSM_src

# Rule to build all files generated by this target.
src/CMakeFiles/ICES__home_manu_robocomp_components_ProyectoMM_Sistemas-Multimedia_ComponentSM_src.dir/build: ICES__home_manu_robocomp_components_ProyectoMM_Sistemas-Multimedia_ComponentSM_src

.PHONY : src/CMakeFiles/ICES__home_manu_robocomp_components_ProyectoMM_Sistemas-Multimedia_ComponentSM_src.dir/build

src/CMakeFiles/ICES__home_manu_robocomp_components_ProyectoMM_Sistemas-Multimedia_ComponentSM_src.dir/clean:
	cd /home/manu/robocomp/components/ProyectoMM/Sistemas-Multimedia/ComponentSM/src && $(CMAKE_COMMAND) -P CMakeFiles/ICES__home_manu_robocomp_components_ProyectoMM_Sistemas-Multimedia_ComponentSM_src.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/ICES__home_manu_robocomp_components_ProyectoMM_Sistemas-Multimedia_ComponentSM_src.dir/clean

src/CMakeFiles/ICES__home_manu_robocomp_components_ProyectoMM_Sistemas-Multimedia_ComponentSM_src.dir/depend:
	cd /home/manu/robocomp/components/ProyectoMM/Sistemas-Multimedia/ComponentSM && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/manu/robocomp/components/ProyectoMM/Sistemas-Multimedia/ComponentSM /home/manu/robocomp/components/ProyectoMM/Sistemas-Multimedia/ComponentSM/src /home/manu/robocomp/components/ProyectoMM/Sistemas-Multimedia/ComponentSM /home/manu/robocomp/components/ProyectoMM/Sistemas-Multimedia/ComponentSM/src /home/manu/robocomp/components/ProyectoMM/Sistemas-Multimedia/ComponentSM/src/CMakeFiles/ICES__home_manu_robocomp_components_ProyectoMM_Sistemas-Multimedia_ComponentSM_src.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/ICES__home_manu_robocomp_components_ProyectoMM_Sistemas-Multimedia_ComponentSM_src.dir/depend
