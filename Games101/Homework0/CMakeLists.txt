cmake_minimum_required( VERSION 3.16.1 ) # Latest version of CMake when this file was created.

set( TARGET_NAME 00-Transformation )

set(HEADER_FILES
)

set(SOURCE_FILES
	src/main.cpp
)

#project( ${TARGET_NAME} )

# Add an executable with the above sources
add_executable( ${TARGET_NAME} 
	${HEADER_FILES}
	${SOURCE_FILES}
	)

target_include_directories( ${TARGET_NAME}
    PUBLIC ../../Extern
)

target_link_libraries( ${TARGET_NAME}
)