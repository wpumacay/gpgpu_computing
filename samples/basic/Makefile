

OPENCV_LIBPATH=/usr/local/lib/
OPENCV_INCLUDEPATH=/usr/local/include/

OPENCV_LIBS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

all: julia vectorAdd longVectorAdd vectorDotProduct

julia: julia_fractal.cu ../helpers/LCommon.cpp Makefile
	nvcc -o julia.out julia_fractal.cu ../helpers/LCommon.cpp -lm -L$(OPENCV_LIBPATH) $(OPENCV_LIBS)

vectorAdd: vectorAdd.cu Makefile
	nvcc -o vectorAdd.out vectorAdd.cu

longVectorAdd: longVectorAdd.cu Makefile
	nvcc -o longVectorAdd.out longVectorAdd.cu

vectorDotProduct: vectorDotProduct.cu Makefile
	nvcc -o vectorDotProduct.out vectorDotProduct.cu

clean: 
	rm *.out
