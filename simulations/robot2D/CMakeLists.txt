cmake_minimum_required(VERSION 3.5.0)
project( particle2D )

find_package( CUDA )
find_package( OpenGL REQUIRED )
find_package( GLEW REQUIRED )

find_package( PkgConfig REQUIRED )
pkg_search_module( GLFW REQUIRED glfw3 )

set( CMAKE_CXX_STANDARD 11 )

include_directories( gl/ ${GLFW_INCLUDE_DIRS} )
link_directories( /usr/lib/ /usr/lib64/ /usr/local/lib/ )

#file( GLOB  CUDA_FILES "/itsvnd/intensification/" *.cu)

set( SOURCES 
        engine/gl/LApp.cpp
        engine/gl/LShaderManager.cpp
        )

#cuda_add_library( VND_LIB SHARED itsvnd/intensification/cuda_vnd.cu )

#cuda_compile( HELPER_CUDA_VND itsvnd/intensification/cuda_vnd.cu )

cuda_add_executable( cs_ds_project 
	        main.cpp 
                gl/LApp.cpp
                gl/LShaderManager.cpp
                )

target_link_libraries( cs_ds_project ${OPENGL_LIBRARIES} ${GLFW_LIBRARIES} GLEW::GLEW )