

#include <iostream>

#include "../helpers/LCommon.h"

using namespace std;


#define N 10000000

#define THREADS_PER_BLOCK 128



__global__ void kernelVectorAdd( float* d_v1, float* d_v2, float* d_v3 )
{
	// note: this is actually less than blockDim.x * gridDim.x
	int tIndx = threadIdx.x + blockIdx.x * blockDim.x;

	while ( tIndx < N )
	{
		d_v3[tIndx] = d_v1[tIndx] + d_v2[tIndx];
		tIndx += blockDim.x * gridDim.x;
	}
}



int main()
{

	float* h_v1 = new float[N];
	float* h_v2 = new float[N];
	float* h_v3 = new float[N];

	for ( int q = 0; q < N; q++ )
	{
		h_v1[q] = ( (float) 2 * q + 1 );
		h_v2[q] = ( (float) 2 * q + 2 );
	}

	float* d_v1;
	float* d_v2;
	float* d_v3;

	cudaMalloc( ( void** ) &d_v1, sizeof( float ) * N );
	cudaMalloc( ( void** ) &d_v2, sizeof( float ) * N );
	cudaMalloc( ( void** ) &d_v3, sizeof( float ) * N );

	cudaMemcpy( d_v1, h_v1, sizeof( float ) * N, cudaMemcpyHostToDevice );
	cudaMemcpy( d_v2, h_v2, sizeof( float ) * N, cudaMemcpyHostToDevice );


	kernelVectorAdd<<< 128, 128 >>>( d_v1, d_v2, d_v3 );

	cudaMemcpy( h_v3 , d_v3, sizeof( float ) * N, cudaMemcpyDeviceToHost );

	free( h_v1 );
	free( h_v2 );
	free( h_v3 );
	cudaFree( d_v1 );
	cudaFree( d_v2 );
	cudaFree( d_v3 );

	return 0;
}