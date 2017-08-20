
#include "ImageHandling.h"


#include <iostream>
#include <string>

using namespace std;

#ifdef USE_CUDA

void rgba_to_greyscale( uchar4* d_rgbaImage,
                        u8* d_greyImage,
                        size_t nRows, size_t nCols );
#elif USE_OPENCL

void rgba_to_greyscale( CEnvironment &env,
                        cl::Buffer* d_rgbaBuff,
                        cl::Buffer* d_greyBuff,
                        u8** h_greyImage,
                        size_t nRows, size_t nCols );

#endif


int main()
{

    CEnvironment _env;

    #ifdef USE_CUDA
        // Do cuda specific initializations if necessary.
        // For now, just using the default
    #elif USE_OPENCL
        // Initialize the CL environment
        vector<cl::Platform> v_platforms;
        cl::Platform::get( &v_platforms );

        cl::Platform _platform = v_platforms.front();
        
        vector<cl::Device> v_devices;
        _platform.getDevices( CL_DEVICE_TYPE_ALL, &v_devices );

        cl::Device _device = v_devices.front();

        cl::Context _context( _device );

        _env.platform = _platform;
        _env.device = _device;
        _env.context = _context;
        cout << "initialized context" << endl;
     #endif

    string _input_file_path( "cinque_terre_small.jpg" );
    
    // Host data
    uchar4 *h_rgbaImage;
    u8 *h_greyImage;

    // Device data
    #ifdef USE_CUDA
    uchar4 *d_rgbaImage;
    u8 *d_greyImage;
    #elif USE_OPENCL
    cl::Buffer d_rgbaBuff;
    cl::Buffer d_greyBuff;
    #endif

    size_t _nRows, _nCols;

    cout << "pre-processing" << endl;

    #ifdef USE_CUDA

    IH::preProcess( _env,
                    &h_rgbaImage, &h_greyImage,
                    &d_rgbaImage, &d_greyImage,
                    _nRows, _nCols,
                    _input_file_path );

    #elif USE_OPENCL
    
    IH::preProcess( _env,
                    &h_rgbaImage, &h_greyImage,
                    &d_rgbaBuff, &d_greyBuff,
                    _nRows, _nCols,
                    _input_file_path );   

    #endif

    cout << "finished pre-processing" << endl;

    #ifdef USE_CUDA

    rgba_to_greyscale( d_rgbaImage, d_greyImage, _nRows, _nCols );
   
    #elif USE_OPENCL

    rgba_to_greyscale( _env, &d_rgbaBuff, &d_greyBuff, &h_greyImage, _nRows, _nCols );

    #endif
    
    #ifdef USE_CUDA
    IH::postProcess( _nRows, _nCols, h_greyImage, d_greyImage );
    #elif USE_OPENCL
    IH::postProcess( _nRows, _nCols, h_greyImage, NULL );
    #endif    

    return 0;
}
