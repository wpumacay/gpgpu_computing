// LCommon.h
// Declarations of some helper functions
#pragma once

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

namespace common
{

	void printDeviceProps( int deviceId = -1 );

	void checkError( cudaError_t pErrorCode );

	template<class T>
	void printArray( T* pArr, int pSize )
	{
		cout << "[ ";
		for ( int q = 0; q < pSize; q++ )
		{
			cout << pArr[q] << " ";
		}
		cout << "]" << endl;
	}
}

