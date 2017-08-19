
#include "ImageHandling.h"


#include <iostream>
#include <string>

#ifdef USE_CUDA
    #include <cuda.h>
    #include <cuda_runtime.h>
#endif


using namespace std;


void rgba_to_greyscale( const uchar4* h_rgbaImage,
                        uchar4* d_rgbaImage,
                        u8* d_greyImage,
                        size_t nRows, size_t nCols );


int main()
{

    string _input_file( "cinque_terre_small.jpg" );
    
    uchar4 *h_rgbaImage, *d_rgbaImage;
    u8 *h_greyImage, *d_greyImage;
    
    size_t _nRows, _nCols;

    IH::preProcess( &h_rgbaImage, &h_greyImage,
                    &d_rgbaImage, &d_greyImage,
                    _nRows, _nCols,
                    _input_file );

    rgba_to_greyscale( h_rgbaImage, d_rgbaImage, d_greyImage, _nRows, _nCols );
   
    size_t _numPixels = _nRows * _nCols;

    cudaMemcpy( h_greyImage, d_greyImage, 
                sizeof( u8 ) * _numPixels,
                cudaMemcpyDeviceToHost );
 
    IH::postProcess( _nRows, _nCols, h_greyImage );

    return 0;
}
