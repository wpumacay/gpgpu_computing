// add.cu
// Add two number example

#include <iostream>

__global__ void add( int a, int b, int *pRes )
{
    *pRes = a + b;
}

int main()
{
    // Create some pointers for GPU memory and cpu memory
    int *d_res;
    int *h_res;

    // Allocate some memory in the GPU, just one int for our result 
    cudaMalloc( ( void** )&d_res, sizeof( int ) );

    // Also, allocate memory for the result, to get it back from our GPU
    h_res = ( int* ) malloc( sizeof( int ) );

    // Call our add "function" ( function, but fancy named "kernel" )
    add<<< 1, 1 >>>( 2, 7, d_res );

    // After the computation, get back the result
    cudaMemcpy( h_res, d_res, sizeof( int ), cudaMemcpyDeviceToHost );
    
    // Print our result
    std::cout << "2 + 7 = " << *h_res << std::endl;

    // Free the memory on the host
    free( h_res );
    // Free the memory on the device ( GPU )
    cudaFree( d_res );
}