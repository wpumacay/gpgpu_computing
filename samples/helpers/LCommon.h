// LCommon.h
// Declarations of some helper functions
#pragma once

#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

typedef unsigned char u8;

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

        template<class T>
        bool areArraysEqual( T* pArr1, T* pArr2, int pSize )
        {
            for ( int q = 0; q < pSize; q++ )
            {
                if ( pArr1[q] != pArr2[q] )
                {
                    return false;
                }
            }

            return true;
        }
	
}

