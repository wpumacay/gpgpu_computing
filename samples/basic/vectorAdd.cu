
#include <iostream>

#include "../helpers/LCommon.h"

using namespace std;

#define VECT_SIZE 100

__global__ void vectorAdd( float* v1, float* v2, float* v3 )
{
	int tIndx = blockIdx.x;

	if ( tIndx < VECT_SIZE )
	{
		v3[tIndx] = v1[tIndx] + v2[tIndx];
	}

}



int main()
{

	// Create our vector pointers
	float* h_v1; float* h_v2; float* h_v3;
	float* d_v1; float* d_v2; float* d_v3;

	// Initialize space for some vectors
	h_v1 = new float[VECT_SIZE];
	h_v2 = new float[VECT_SIZE];
	h_v3 = new float[VECT_SIZE];

	cudaMalloc( ( void** ) &d_v1, sizeof( float ) * VECT_SIZE );
	cudaMalloc( ( void** ) &d_v2, sizeof( float ) * VECT_SIZE );
	cudaMalloc( ( void** ) &d_v3, sizeof( float ) * VECT_SIZE );

	// Initialize the host vectors with some values
	for ( int q = 0; q < VECT_SIZE; q++ )
	{
		h_v1[q] = 2 * q + 1;
		h_v2[q] = 2 * q + 2;
	}

	// Copy the working data to GPU
	cudaMemcpy( d_v1, h_v1, sizeof( float ) * VECT_SIZE, cudaMemcpyHostToDevice );
	cudaMemcpy( d_v2, h_v2, sizeof( float ) * VECT_SIZE, cudaMemcpyHostToDevice );

	// Call our vectorAdd kernel
	vectorAdd<<< 1, VECT_SIZE >>>( d_v1, d_v2, d_v3 );

	// Get the data back from GPU
	cudaMemcpy( h_v3, d_v3, sizeof( float ) * VECT_SIZE, cudaMemcpyDeviceToHost );

	// Print the results
	common::printArray( h_v3, VECT_SIZE );

	// Free the resources
	delete[] h_v1;
	delete[] h_v2;
	delete[] h_v3;

	cudaFree( d_v1 );
	cudaFree( d_v2 );
	cudaFree( d_v3 );

	return 0;

}