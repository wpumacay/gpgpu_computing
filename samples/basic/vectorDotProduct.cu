

#include <iostream>

using namespace std;


#define N 10000000
#define THREADS_PER_BLOCK 128
#define BLOCKS_PER_GRID 32

__global__ void kernelVectorDot( float* d_v1, float* d_v2, float* d_partials )
{
	// note: this is actually less than blockDim.x * gridDim.x
	int tIndx = threadIdx.x + blockIdx.x * blockDim.x;
	int shIndx = threadIdx.x;

	__shared__ float _sh_results[THREADS_PER_BLOCK];

	float _tmp = 0.0f;

	while ( tIndx < N )
	{
		_tmp = d_v1[tIndx] * d_v2[tIndx];
		tIndx += blockDim.x * gridDim.x;
	}

	_sh_results[shIndx] = _tmp;

	__syncthreads();

	// Reduction among the threads in a block
	int i = blockDim.x / 2;
	while ( i != 0 )
	{
		if ( shIndx < i )
		{
			_sh_results[shIndx] = _sh_results[shIndx + i];
		}
		__syncthreads();
		i /= 2;
	}

	if ( shIndx == 0 )
	{
		d_partials[blockIdx.x] = _sh_results[0];
	}
}


int main()
{

	float* h_v1 = new float[N];
	float* h_v2 = new float[N];
	float* h_partials = new float[BLOCKS_PER_GRID];

	for ( int q = 0; q < N; q++ )
	{
		h_v1[q] = ( (float) 2 * q + 1 );
		h_v2[q] = ( (float) 2 * q + 2 );
	}

	for ( int q = 0; q < BLOCKS_PER_GRID; q++ )
	{
		h_partials[q] = 0.0f;
	}

	float* d_v1;
	float* d_v2;
	float* d_partials;

	cudaMalloc( ( void** ) &d_v1, sizeof( float ) * N );
	cudaMalloc( ( void** ) &d_v2, sizeof( float ) * N );
	cudaMalloc( ( void** ) &d_partials, sizeof( float ) * BLOCKS_PER_GRID );

	cudaMemcpy( d_v1, h_v1, sizeof( float ) * N, cudaMemcpyHostToDevice );
	cudaMemcpy( d_v2, h_v2, sizeof( float ) * N, cudaMemcpyHostToDevice );
	cudaMemcpy( d_partials, h_partials, sizeof( float ) * BLOCKS_PER_GRID, cudaMemcpyHostToDevice );
	

	kernelVectorDot<<< BLOCKS_PER_GRID, THREADS_PER_BLOCK >>>( d_v1, d_v2, d_partials );


	cudaMemcpy( h_partials, d_partials, sizeof( float ) * BLOCKS_PER_GRID, cudaMemcpyDeviceToHost );

	float _sum = 0.0f;
	for ( int q = 0; q < BLOCKS_PER_GRID; q++ )
	{
		_sum += h_partials[q];
	}

	cout << "dot: " << _sum << endl;

	free( h_v1 );
	free( h_v2 );
	free( h_partials );
	cudaFree( d_v1 );
	cudaFree( d_v2 );
	cudaFree( d_partials );
}