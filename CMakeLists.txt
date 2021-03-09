cmake_minimum_required( VERSION 3.16.1 ) # Latest version of CMake when this file was created.

# Make sure DLL and EXE targets go to the same directory.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib) # Output directory for static lib (.LIB)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) # Output directory for shared lib (.DLL)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin) # Output directory for executables (.EXE)

project( JZ_Games101 )
add_subdirectory(Games101/Homework0)
add_subdirectory(Games101/Homework1)
add_subdirectory(Games101/Homework2)
add_subdirectory(Games101/Homework3)