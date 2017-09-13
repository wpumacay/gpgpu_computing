cmake_minimum_required(VERSION 3.5.0)
project( particle2D )

set(CMAKE_BUILD_TYPE Debug)

##find_package( CUDA )
find_package( OpenGL REQUIRED )
find_package( GLEW REQUIRED )

find_package( PkgConfig REQUIRED )
pkg_search_module( GLFW REQUIRED glfw3 )

set( CMAKE_CXX_STANDARD 11 )

include_directories( gl/ app/ ${GLFW_INCLUDE_DIRS} )
link_directories( /usr/lib/ /usr/lib64/ /usr/local/lib/ )

#file( GLOB  CUDA_FILES "/itsvnd/intensification/" *.cu)

set( SOURCES 
     gl/LApp.cpp
     gl/core/primitives/LPrimitivesRenderer2D.cpp
     gl/core/shaders/LShaderManager.cpp
     app/graphics/LAppGraphics2D.cpp
     app/robotics/LAppRobotics2D.cpp )

#cuda_add_library( VND_LIB SHARED itsvnd/intensification/cuda_vnd.cu )

#cuda_compile( HELPER_CUDA_VND itsvnd/intensification/cuda_vnd.cu )

#cuda_add_executable( cs_ds_project 
#	        main.cpp 
#                gl/LApp.cpp
#                gl/LShaderManager.cpp
#                )

add_executable( particle2D
				main.cpp
				gl/LApp.cpp
				gl/core/primitives/LPrimitivesRenderer2D.cpp
				gl/core/shaders/LShaderManager.cpp
				app/graphics/LAppGraphics2D.cpp
				app/robotics/LAppRobotics2D.cpp )

target_link_libraries( particle2D ${OPENGL_LIBRARIES} ${GLFW_LIBRARIES} GLEW::GLEW )