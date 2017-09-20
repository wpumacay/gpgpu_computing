
#pragma once

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>



namespace common
{


	void printDeviceProps( int deviceId = -1 );

	void checkError( cudaError_t pErrorCode );




}

