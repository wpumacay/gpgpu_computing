
#include "LCommon.h"

using namespace std;

namespace common
{


	void printDeviceProps( int deviceId )
	{
		if ( deviceId == -1 )
		{
			int _devCount;
			// Check all devices
			checkError( cudaGetDeviceCount( &_devCount ) );

			for ( int q = 0; q < _devCount; q++ )
			{
				printDeviceProps( q );
			}
		}
		else
		{
			cudaDeviceProp _props;

			checkError( cudaGetDeviceProperties( &_props, deviceId ) );

			cout << " --- General information for device " << deviceId << " ---" << endl;
			cout << "Name: " << _props.name;
			cout << "Compute capability: " << _props.major << "." << _props.minor << endl;
			cout << "Clock rate: " << _props.clockRate << endl;
			cout << "Device copy overlap: " << ( ( _props.deviceOverlap ) ? "Enabled" : "Disabled" ) << endl;
			cout << "Kernel execution timeout: " << ( ( _props.kernelExecTimeoutEnabled ) ? "Enabled" : "Disabled" ) << endl;

			cout << " --- Memory information for device " << deviceId << " ---" << endl;
			cout << "Total global memory: " << _props.totalGlobalMem << endl;
			cout << "Total constant memory: " << _props.totalConstMem << endl;
			cout << "Max mem pitch: " << _props.memPitch << endl;
			cout << "Texture Alignment: " << _props.textureAlignment << endl;

			cout << " --- Multiprocessor information for device " << deviceId << " ---" << endl;
			cout << "Num multiprocessors: " << _props.multiProcessorCount << endl;
			cout << "Shared memory per multiprocessor: " << _props.sharedMemPerBlock << endl;
			cout << "Registers per multiprocessor: " << _props.regsPerBlock << endl;
			cout << "Threads in warp: " << _props.warpSize << endl;

			cout << "Max threads per block: " << _props.maxThreadsPerBlock << endl;
			cout << "Max thread dimensions: " << "( " 
					<< _props.maxThreadsDim[0] << " - "
					<< _props.maxThreadsDim[1] << " - "
					<< _props.maxThreadsDim[2] << " ) " << endl;
			cout << "Max grid dimensions: " << "( " 
					<< _props.maxGridSize[0] << " - "
					<< _props.maxGridSize[1] << " - "
					<< _props.maxGridSize[2] << " ) " << endl;
			cout << " ------------------------------------------------------------ " << endl;
		}
	}


	void checkError( cudaError_t pErrorCode )
	{
		if ( pErrorCode != cudaSuccess )
		{
			cerr << "Error: " << __FILE__ << " - " << __LINE__ << endl;
			cerr << "Code: " << pErrorCode << " - reason: " << cudaGetErrorString( pErrorCode ) << endl;

			exit( 1 );
		}
	}


}
