
#include "LopenclHelpers.h"

namespace clUtils
{

    string loadKernelSrc( const string &path )
    {
        string _src;
        
        ifstream _file( path.c_str() );
        
        string _line;

        while( getline( _file, _line ) )
        {
            _src += _line;
        }

        return _src;
    }

    
    cl::Program createProgram( const cl::Context &context,
                               const cl::Device &device,
                               const string& pathToKernelSrc )
    {
        string _strKernelSrc = loadKernelSrc( pathToKernelSrc );

        cl::Program::Sources _kernelSrc( 1, 
                                         make_pair( _strKernelSrc.c_str(),
                                                    _strKernelSrc.size() + 1 ) );
        
        cl::Program _program( context, _kernelSrc );
        cl_int _err = _program.build( "-cl-std=CL1.2" );

        if ( _err != CL_SUCCESS )
        {
            cout << "build info: ******" << endl;

            cout << _program.getBuildInfo<CL_PROGRAM_BUILD_LOG>( device ) << endl;

            cout << "******************" << endl;
        }

        return _program;
    }


}





void cl_rb_pf_motion_model_step( cl::Program& program,
                                 cl::Context& context,
                                 cl::Device& device,
                                 ClParticle* h_particles, int nParticles, 
                                 float dt, float v, float w )
{

	float ddt = dt;
	float vv = v;
	float ww = w;

    cl_int _err;
    cl::Kernel _kernel( program, "rb_update_particle", &_err );

    cl::Buffer d_buff_particles( context,
                                 CL_MEM_READ_WRITE |
                                 CL_MEM_COPY_HOST_PTR,
                                 sizeof( ClParticle ) * nParticles,
                                 h_particles );

    _err = _kernel.setArg( 0, d_buff_particles );
    _err = _kernel.setArg( 1, sizeof( int ), &nParticles );
    _err = _kernel.setArg( 2, sizeof( float ), &ddt );
    _err = _kernel.setArg( 3, sizeof( float ), &vv );
    _err = _kernel.setArg( 4, sizeof( float ), &ww );

    cl::CommandQueue _cmd_queue( context, device );
    _err = _cmd_queue.enqueueNDRangeKernel( _kernel,
                                            cl::NullRange,
                                            cl::NDRange( nParticles ) );

    _err = _cmd_queue.enqueueReadBuffer( d_buff_particles, CL_TRUE,
                                         0, sizeof( ClParticle ) * nParticles,
                                         h_particles );
}

void cl_rb_pf_sensor_model_step( cl::Program& program,
                              	 cl::Context& context,
                              	 cl::Device& device,
                              	 ClParticle* h_particles, int nParticles, 
                              	 ClLine* h_lines, int nLines,
                              	 float* h_sensorsZ, float* h_sensorsAng, int nSensors )
{

	int _nParticles = nParticles;
	int _nLines = nLines;
	int _nSensors = nSensors;

    cl_int _err;
    cl::Kernel _kernel( program, "rb_raycast_particle", &_err );
    cl::Kernel _kernel_2( program, "rb_update_particle_weight", &_err );

    cl::Buffer d_buff_particles( context,
                                 CL_MEM_READ_WRITE |
                                 CL_MEM_COPY_HOST_PTR,
                                 sizeof( ClParticle ) * nParticles,
                                 h_particles );

    cl::Buffer d_buff_lines( context,
    						 CL_MEM_READ_WRITE |
                             CL_MEM_COPY_HOST_PTR,
                             sizeof( ClLine ) * nLines,
                             h_lines );

    cl::Buffer d_buff_sensorsZ( context,
    							CL_MEM_READ_WRITE |
                                CL_MEM_COPY_HOST_PTR,
                                sizeof( float ) * nSensors,
                                h_sensorsZ );

    cl::Buffer d_buff_sensorsAng( context,
    							  CL_MEM_READ_WRITE |
                                  CL_MEM_COPY_HOST_PTR,
                                  sizeof( float ) * nSensors,
                                  h_sensorsAng );

    cl::CommandQueue _cmd_queue( context, device );

	for ( int l = 0; l < nLines; l++ )
	{
		for ( int s = 0; s < nSensors; s++ )
		{
		    _kernel.setArg( 0, d_buff_particles );
		    _kernel.setArg( 1, sizeof( int ), &_nParticles );
		    _kernel.setArg( 2, d_buff_lines );
		    _kernel.setArg( 3, sizeof( int ), &_nLines );
		    _kernel.setArg( 4, d_buff_sensorsZ );
		    _kernel.setArg( 5, d_buff_sensorsAng );
		    _kernel.setArg( 6, sizeof( int ), &_nSensors );
		    _kernel.setArg( 7, sizeof( int ), &l );
		    _kernel.setArg( 8, sizeof( int ), &s );

		    _cmd_queue.enqueueNDRangeKernel( _kernel,
		                                     cl::NullRange,
		                                     cl::NDRange( nParticles ) );

//		    _cmd_queue.enqueueReadBuffer( d_buff_particles, CL_TRUE,
//		                                  0, sizeof( ClParticle ) * nParticles,
//		                                  h_particles );
		}
	}

	_kernel_2.setArg( 0, d_buff_particles );
	_kernel_2.setArg( 1, sizeof( int ), &_nParticles );
	_kernel_2.setArg( 2, d_buff_sensorsZ );
	_kernel_2.setArg( 3, sizeof( int ), &_nSensors );

	_cmd_queue.enqueueNDRangeKernel( _kernel_2,
		                             cl::NullRange,
		                             cl::NDRange( nParticles ) );

	_cmd_queue.enqueueReadBuffer( d_buff_particles, CL_TRUE,
		                          0, sizeof( ClParticle ) * nParticles,
		                          h_particles );
}

