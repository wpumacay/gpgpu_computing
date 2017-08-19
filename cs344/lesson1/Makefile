NVCC=nvcc

###################################
# These are the default install   #
# locations on most linux distros #
###################################

OPENCV_LIBPATH=/usr/local/lib/
OPENCV_INCLUDEPATH=/usr/local/include/

OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

CUDA_INCLUDEPATH=/usr/local/cuda-8.0/include/

solution: main.o cuda_handlers.o Makefile
	$(NVCC) -o main.out main.o cuda_handlers.o -lm -L$(OPENCV_LIBPATH) $(OPENCV_LIBS)

main.o: main.cpp ImageHandling.h 
	g++ -c main.cpp -I$(CUDA_INCLUDEPATH) -I$(OPENCV_INCLUDEPATH)

cuda_handlers.o: cuda_handlers.cu
	nvcc -c cuda_handlers.cu

clean:
	rm -f *.o *.png hw
