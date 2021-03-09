cmake_minimum_required( VERSION 3.16.1 ) # Latest version of CMake when this file was created.

set( TARGET_NAME 06-WhittedStyleRayTracing-BVH)

set(HEADER_FILES
)

set(SOURCE_FILES
	src/main.cpp
	src/Object.hpp 
	src/Vector.hpp
	src/Vector.cpp 
	src/Sphere.hpp 
	src/global.hpp
	src/Triangle.hpp
	src/Scene.hpp
	src/Scene.cpp
	src/Light.hpp
	src/AreaLight.hpp
	src/BVH.cpp
	src/BVH.hpp
	src/Bounds3.hpp
	src/Ray.hpp
	src/Material.hpp
	src/Intersection.hpp
	src/Renderer.hpp
	src/Renderer.cpp
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