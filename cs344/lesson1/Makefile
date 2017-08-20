

###################################
# These are the default install   #
# locations on most linux distros #
###################################

OPENCV_LIBPATH=/usr/local/lib/
OPENCV_INCLUDEPATH=/usr/local/include/

OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs
OPENCL_LIBS=-lOpenCL

CUDA_INCLUDEPATH=/usr/local/cuda-8.0/include/
CL_INCLUDEPATH=/usr/include/CL/

cuda: main_cuda.o cuda_handlers.o Makefile
	nvcc -o lesson1_cuda.out main_cuda.o cuda_handlers.o -lm -L$(OPENCV_LIBPATH) $(OPENCV_LIBS)

opencl: main_opencl.o cl_handlers.o Makefile
	g++ -g -o lesson1_cl.out main_opencl.o cl_handlers.o -lm -L$(OPENCV_LIBPATH) $(OPENCV_LIBS) $(OPENCL_LIBS)

main_cuda.o: main.cpp ImageHandling.h 
	g++ -c -o main_cuda.o main.cpp -DUSE_CUDA -I$(CUDA_INCLUDEPATH) -I$(OPENCV_INCLUDEPATH)

main_opencl.o: main.cpp ImageHandling.h
	g++ -DUSE_OPENCL -c -g -o main_opencl.o main.cpp -I$(CL_INCLUDEPATH) -I$(OPENCV_INCLUDEPATH)

cl_handlders.o: cl_handlers.cpp utils.h Common.h
	g++ -DUSE_OPENCL -c -g cl_handlers.cpp -I$(CL_INCLUDEPATH)

cuda_handlers.o: cuda_handlers.cu
	nvcc -c cuda_handlers.cu

clean:
	rm -f *.o *.png hw
