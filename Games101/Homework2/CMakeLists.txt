cmake_minimum_required( VERSION 3.16.1 ) # Latest version of CMake when this file was created.

set( TARGET_NAME 02-Rasterization)

set(HEADER_FILES
)

set(SOURCE_FILES
	src/main.cpp
	src/rasterizer.hpp 
	src/rasterizer.cpp 
	src/Triangle.hpp 
	src/Triangle.cpp
	src/global.hpp
)

#project( ${TARGET_NAME} )

# Add static lib path 
link_directories(../../Extern/OpenCV/lib/Win64/)

# Add an executable with the above sources
add_executable( ${TARGET_NAME} 
	${HEADER_FILES}
	${SOURCE_FILES}
	)

target_include_directories( ${TARGET_NAME}
    PUBLIC ../../Extern
	PUBLIC ../../Extern/OpenCV/include
	PUBLIC ../../Extern/OpenCV/include/opencv
	PUBLIC ../../Extern/OpenCV/include/opencv2	
)

target_link_libraries( ${TARGET_NAME}
	opencv_world3412d.lib
)
