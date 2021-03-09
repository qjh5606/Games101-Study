cmake_minimum_required( VERSION 3.16.1 ) # Latest version of CMake when this file was created.

set( TARGET_NAME 03-RasterizationAdv)

set(HEADER_FILES
)

set(SOURCE_FILES
	src/main.cpp
	src/rasterizer.hpp 
	src/rasterizer.cpp
	src/global.hpp 
	src/Triangle.hpp 
	src/Triangle.cpp
	src/global.hpp
	src/Texture.hpp
	src/Texture.cpp
	src/Shader.hpp
	src/OBJ_Loader.h
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

target_compile_features( ${TARGET_NAME} 
	PUBLIC cxx_std_17
)