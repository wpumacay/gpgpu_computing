
#include "ImageHandling.h"

#include <iostream>
#include <string>

using namespace std;

/** Externs ***********************************************************************************/
void apply_gaussian_blur( uchar4* h_inputImageRGBA, 
                          uchar4* d_inputImageRGBA,
                          uchar4* d_outputImageRGBA, 
                          size_t numRows, size_t numCols,
                          unsigned char* d_red, 
                          unsigned char* d_green, 
                          unsigned char* d_blue,
                          unsigned char* d_redBlurred, 
                          unsigned char* d_greenBlurred, 
                          unsigned char* d_blueBlurred,
                          float* d_filter,
                          int filterDim );

/** *******************************************************************************************/

int main() 
{
    IH::CWorkingImg _ws;
    string input_file( "cinque_terre_small.jpg" );

    //load the image and give us our input and output pointers
    _ws = IH::preProcess( input_file );

    apply_gaussian_blur( _ws.h_inputImageRGBA, 
                         _ws.d_inputImageRGBA,
                         _ws.d_outputImageRGBA, 
                         _ws.rows, _ws.cols,
                         _ws.d_red,
                         _ws.d_green,
                         _ws.d_blue,
                         _ws.d_redBlurred, 
                         _ws.d_greenBlurred, 
                         _ws.d_blueBlurred, 
                         _ws.d_filter,
                         _ws.filterDim );
  
    cudaDeviceSynchronize();

    size_t _numPixels = _ws.rows * _ws.cols;

    cudaMemcpy( _ws.h_outputImageRGBA, 
                _ws.d_outputImageRGBA, 
                sizeof( uchar4 ) * _numPixels, 
                cudaMemcpyDeviceToHost );

    IH::postProcess( _ws.h_outputImageRGBA,
                     _ws.rows, _ws.cols );


    IH::cleanUp( _ws );

    return 0;
}
