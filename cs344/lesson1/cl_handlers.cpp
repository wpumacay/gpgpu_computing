
#include "Common.h"
#include "utils.h"

#include <string>
#include <iostream>

using namespace std;


void rgba_to_greyscale( CEnvironment &env,
                        cl::Buffer* d_rgbaBuff,
                        cl::Buffer* d_greyBuff,
                        u8** h_greyImage,
                        size_t nRows, size_t nCols )
{
    cl::Program _program = utils::createProgram( env.context,
                                                 env.device,
                                                 string( "cl_kernels.cl" ) );
    
    cl_int _err;
    
    cl::Kernel _kernel( _program, "kernel_rgb2grey", &_err );

    cout << "err? " << _err << endl;

    int _nRows = nRows;
    int _nCols = nCols;

    _err = _kernel.setArg( 0, *d_rgbaBuff );
    cout << "err? " << _err << endl;
    _err = _kernel.setArg( 1, *d_greyBuff );
    cout << "err? " << _err << endl;
    _err = _kernel.setArg( 2, sizeof( int ), &_nRows  );
    cout << "err? " << _err << endl;
    _err = _kernel.setArg( 3, sizeof( int ), &_nCols ); 
    cout << "err? " << _err << endl;

    cl::CommandQueue _cmd_queue( env.context, env.device );
    _cmd_queue.enqueueNDRangeKernel( _kernel, 
                                     cl::NullRange,
                                     cl::NDRange( nCols, nRows ),
                                     cl::NDRange( 10, 10 ) );
    
    size_t _numPixels = nRows * nCols;
    _cmd_queue.enqueueReadBuffer( *d_greyBuff,
                                  CL_TRUE,
                                  0, sizeof( u8 ) * _numPixels,
                                  *h_greyImage );
}


